import streamlit as st
import os, json, numpy as np, yfinance as yf, pickle, faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from rank_bm25 import BM25Okapi

# ============================================================================
# 1. CONFIGURATION & PROMPTS (Ideally, move this to a separate config.py)
# ============================================================================
SYSTEM_PROMPT = """You are a financial analysis expert. Use ReAct: 1. THOUGHT 2. ACTION 3. OBSERVATION. Tools: tool_rag_search, tool_get_stock_data, tool_calculate, tool_financial_ratios. Do not hallucinate data."""

# Abstracted tools list to reduce file size
TOOLS = [
    {"type": "function", "function": {"name": "tool_rag_search", "description": "Search 10-K report", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "tool_get_stock_data", "description": "Get live stock metrics", "parameters": {"type": "object", "properties": {"ticker": {"type": "string", "default": "AAPL"}}}}},
    {"type": "function", "function": {"name": "tool_calculate", "description": "Perform math", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "tool_financial_ratios", "description": "Calculate ratios (pe_ratio, profit_margin, roe, debt_to_equity, current_ratio)", "parameters": {"type": "object", "properties": {"metric": {"type": "string"}, "value1": {"type": "number"}, "value2": {"type": "number"}}, "required": ["metric", "value1"]}}}
]

# ============================================================================
# 2. CORE FUNCTIONS & TOOLS 
# ============================================================================
@st.cache_resource
def init_system():
    """Initializes models and data once to save memory and reduce boilerplate."""
    try:
        embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        cross_enc = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        idx_path = 'index.faiss' if os.path.exists('index.faiss') else 'index (1).faiss'
        faiss_idx = faiss.read_index(idx_path) if os.path.exists(idx_path) else None
        
        with open('apple_documents.pkl', 'rb') as f: docs = pickle.load(f)
        meta = [{'text': getattr(d, 'page_content', str(d)), 'metadata': getattr(d, 'metadata', {})} for d in docs]
        bm25 = BM25Okapi([d['text'].lower().split() for d in meta])
        
        return embedder, cross_enc, faiss_idx, meta, bm25
    except Exception as e:
        st.error(f"Initialization Error: {e}")
        return [None]*5

def execute_tool(name: str, args: dict) -> str:
    """Consolidated tool execution router."""
    try:
        if name == "tool_get_stock_data":
            info = yf.Ticker(args.get('ticker', 'AAPL')).info
            return json.dumps({k: info.get(k, 'N/A') for k in ['currentPrice', 'marketCap', 'trailingPE', 'forwardPE']}, indent=2)
        elif name == "tool_calculate":
            return str(eval(args['expression'], {"__builtins__": {}}, {"abs": abs, "round": round, "sum": sum, "min": min, "max": max}))
        elif name == "tool_financial_ratios":
            m, v1, v2 = args['metric'].lower(), args['value1'], args.get('value2', 1)
            calcs = {"pe_ratio": v1/v2, "profit_margin": (v1/v2)*100, "roe": (v1/v2)*100, "debt_to_equity": v1/v2, "current_ratio": v1/v2}
            return f"{m}: {calcs.get(m, 'Unknown metric')}"
        elif name == "tool_rag_search":
            return run_hybrid_search(args['query'], args.get('top_k', 5))
        return f"Tool {name} not implemented."
    except Exception as e:
        return f"Tool error: {e}"

def run_hybrid_search(query: str, top_k: int) -> str:
    """Extracted search logic for cleanliness."""
    emb, cross, idx, meta, bm25 = [st.session_state.get(k) for k in ['emb', 'cross', 'idx', 'meta', 'bm25']]
    if not all([emb, idx, meta, bm25]): return "RAG not initialized"
    
    q_emb = np.array(emb.encode([query]), dtype=np.float32).reshape(1, -1)
    dists, indices = idx.search(q_emb, min(top_k * 2, len(meta)))
    bm25_scores = bm25.get_scores(query.lower().split())
    
    results = {int(i): {'f': 1/(1+float(d)), 'b': float(bm25_scores[int(i)])} for i, d in zip(indices[0], dists[0]) if 0 <= i < len(meta)}
    max_f, max_b = max([v['f'] for v in results.values()] or [1]), max([v['b'] for v in results.values()] or [1])
    
    top_docs = [meta[i] for i, _ in sorted(results.items(), key=lambda x: 0.7*(x[1]['f']/max_f) + 0.3*(x[1]['b']/max_b), reverse=True)[:top_k]]
    ranked = sorted(zip(top_docs, cross.predict([[query, d['text']] for d in top_docs])), key=lambda x: x[1], reverse=True)[:3] if cross else [(d, 0) for d in top_docs]
    
    return "\n\n".join([f"[Context {i+1}]: {ctx['text'][:500]}..." for i, (ctx, _) in enumerate(ranked)])

# ============================================================================
# 3. AGENT LOOP 
# ============================================================================
def react_agent(query: str, client: Groq, max_iter: int = 5) -> dict:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    state = {'thoughts': [], 'actions': [], 'obs': [], 'answer': ''}
    
    for i in range(max_iter):
        res = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, tools=TOOLS, tool_choice="auto", temperature=0.1)
        msg = res.choices[0].message
        
        if not msg.tool_calls:
            state['answer'] = msg.content
            return state
            
        state['thoughts'].append(f"💭 Iteration {i+1}: Tool usage required.")
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            state['actions'].append(f"🔧 {tc.function.name}({args})")
            obs = execute_tool(tc.function.name, args)
            state['obs'].append(f"👁️ {obs[:150]}...")
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
            
    state['answer'] = "Max iterations reached."
    return state

# ============================================================================
# 4. STREAMLIT UI
# ============================================================================
def main():
    st.title("🤖 Agentic Financial RAG")
    api_key = st.secrets.get('GROQ_API_KEY') or st.session_state.get('api_key') or st.text_input("Groq API Key:", type="password")
    if api_key: st.session_state.api_key = api_key
    else: return st.warning("Please enter your Groq API key.")
    
    with st.spinner("Loading..."):
        st.session_state.emb, st.session_state.cross, st.session_state.idx, st.session_state.meta, st.session_state.bm25 = init_system()
    
    query = st.text_area("Ask the Agent:", placeholder="E.g., What was Apple's net income?")
    if st.button("🚀 Run Agent") and query:
        with st.spinner("Thinking..."):
            res = react_agent(query, Groq(api_key=api_key))
            st.markdown(f"### Answer\n{res['answer']}")
            with st.expander("🔍 Reasoning Process"):
                for t, a, o in zip(res['thoughts'], res['actions'], res['obs']):
                    st.info(t); st.code(a); st.success(o)

if __name__ == "__main__":
    main()

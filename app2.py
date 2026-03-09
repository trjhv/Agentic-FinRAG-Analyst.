import streamlit as st
import os, json, pickle, faiss, tempfile, shutil, numpy as np, pandas as pd, yfinance as yf
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# ============================================================================
# CONFIGURATION & TOOLS
# ============================================================================
SYS_PROMPT = "You are a financial analysis expert analyzing {company} ({ticker}). Use ReAct pattern: Think -> Act -> Observe. Be accurate. Use tools."

TOOLS = [
    {"type": "function", "function": {"name": "tool_rag_search", "description": "Search 10-K data", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "tool_get_stock_data", "description": "Get live stock data", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "tool_calculate", "description": "Math calculations", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "tool_financial_ratios", "description": "Calculate ratios (profit_margin, pe_ratio, roe)", "parameters": {"type": "object", "properties": {"metric": {"type": "string"}, "value1": {"type": "number"}, "value2": {"type": "number"}}, "required": ["metric", "value1"]}}},
    {"type": "function", "function": {"name": "tool_compare_companies", "description": "Compare two companies", "parameters": {"type": "object", "properties": {"ticker1": {"type": "string"}, "ticker2": {"type": "string"}}, "required": ["ticker1", "ticker2"]}}}
]

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
@st.cache_data
def load_companies():
    for f in ['company_tickers_RAG.csv', 'fortune500_tickers.csv']:
        if os.path.exists(f): return {row['Company'].strip(): row['Ticker'].strip() for _, row in pd.read_csv(f).iterrows()}
    return {'Apple': 'AAPL', 'Microsoft': 'MSFT', 'Alphabet Inc.': 'GOOGL', 'Amazon': 'AMZN', 'Tesla': 'TSLA'}

@st.cache_resource
def load_models():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2'), CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def load_uploaded_files(f_file, m_file, b_file):
    d = tempfile.mkdtemp()
    paths = [os.path.join(d, n) for n in ['idx.faiss', 'm.pkl', 'b.pkl']]
    for p, f in zip(paths, [f_file, m_file, b_file]):
        with open(p, 'wb') as out: out.write(f.read())
    idx = faiss.read_index(paths[0])
    with open(paths[1], 'rb') as f: meta = pickle.load(f)
    with open(paths[2], 'rb') as f: bm25 = pickle.load(f)
    shutil.rmtree(d)
    return idx, meta, bm25

def run_tools(name: str, args: dict) -> str:
    try:
        if name == "tool_get_stock_data":
            info = yf.Ticker(args.get('ticker') or st.session_state.get('company_ticker', 'AAPL')).info
            return json.dumps({k: info.get(k, 'N/A') for k in ['currentPrice', 'marketCap', 'trailingPE', 'fiftyTwoWeekHigh']}, indent=2)
        elif name == "tool_calculate":
            return str(eval(args['expression'], {"__builtins__": {}}, {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}))
        elif name == "tool_financial_ratios":
            m, v1, v2 = args['metric'].lower(), args['value1'], args.get('value2', 1)
            return f"{m}: {v1/v2 if m == 'pe_ratio' else (v1/v2)*100}"
        elif name == "tool_compare_companies":
            return json.dumps({t: {k: yf.Ticker(t).info.get(k, 'N/A') for k in ['currentPrice', 'trailingPE']} for t in [args['ticker1'], args['ticker2']]})
        elif name == "tool_rag_search":
            emb, cross, idx, meta, bm25 = [st.session_state.get(k) for k in ['emb', 'cross', 'idx', 'meta', 'bm25']]
            if not all([emb, idx, meta, bm25]): return "Data missing."
            q_emb = np.array(emb.encode([args['query']]), dtype=np.float32).reshape(1, -1)
            dists, indices = idx.search(q_emb, 10)
            scores = {int(i): {'f': 1/(1+float(d)), 'b': float(bm25.get_scores(args['query'].lower().split())[int(i)])} for i, d in zip(indices[0], dists[0]) if 0 <= i < len(meta)}
            mf, mb = max([v['f'] for v in scores.values()] or [1]), max([v['b'] for v in scores.values()] or [1])
            top = [meta[i] for i, _ in sorted(scores.items(), key=lambda x: 0.7*(x[1]['f']/mf) + 0.3*(x[1]['b']/mb), reverse=True)[:args.get('top_k', 5)]]
            ranked = sorted(zip(top, cross.predict([[args['query'], d['text']] for d in top])), key=lambda x: x[1], reverse=True)[:3] if cross else [(d, 0) for d in top]
            return f"From {st.session_state.get('selected_company')}:\n" + "\n".join([f"[{i+1}]: {c['text'][:500]}..." for i, (c, _) in enumerate(ranked)])
    except Exception as e: return f"Error: {e}"

def react_agent(query: str, client: Groq, max_iter: int = 5) -> dict:
    msgs = [{"role": "system", "content": SYS_PROMPT.format(company=st.session_state.get('selected_company'), ticker=st.session_state.get('company_ticker'))}, {"role": "user", "content": query}]
    state = {'thoughts': [], 'actions': [], 'obs': [], 'answer': ''}
    
    for i in range(max_iter):
        msg = client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, tools=TOOLS, tool_choice="auto", temperature=0.1).choices[0].message
        if not msg.tool_calls:
            state['answer'] = msg.content
            return state
            
        state['thoughts'].append(f"💭 Iteration {i+1}: Using tools")
        msgs.append({"role": "assistant", "content": msg.content or "", "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            state['actions'].append(f"🔧 {tc.function.name}({args})")
            obs = run_tools(tc.function.name, args)
            state['obs'].append(f"👁️ {obs[:200]}...")
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": obs})
            
    state['answer'] = "Max iterations reached"
    return state

# ============================================================================
# UI
# ============================================================================
def main():
    st.title("🧠 Bring Your Own Brain RAG")
    key = st.secrets.get('GROQ_API_KEY') or st.session_state.get('api_key') or st.text_input("Groq API Key:", type="password")
    if not key: return st.warning("Enter Groq Key")
    st.session_state.api_key = key
    
    st.session_state.emb, st.session_state.cross = load_models()
    comps = load_companies()
    
    col1, col2 = st.columns(2)
    with col1: st.session_state.selected_company = st.selectbox("Company:", sorted(comps.keys()))
    with col2: st.session_state.company_ticker = st.text_input("Ticker:", comps.get(st.session_state.selected_company, 'AAPL')).upper()
    
    c1, c2 = st.columns(2)
    with c1: f_file, b_file = st.file_uploader("FAISS (.faiss)"), st.file_uploader("BM25 (.pkl)")
    with c2: m_file, pdf = st.file_uploader("Metadata (.pkl)"), st.file_uploader("10-K PDF")
    
    if all([f_file, m_file, b_file]) and st.button("🚀 Load Data"):
        with st.spinner("Loading..."):
            st.session_state.idx, st.session_state.meta, st.session_state.bm25 = load_uploaded_files(f_file, m_file, b_file)
            st.success("Data Loaded!")
            
    if 'idx' in st.session_state:
        query = st.text_area("Ask Question:")
        if st.button("Run Agent") and query:
            res = react_agent(query, Groq(api_key=key))
            st.markdown(f"### Answer\n{res['answer']}")
            with st.expander("🔍 Reasoning"):
                for t, a, o in zip(res['thoughts'], res['actions'], res['obs']):
                    st.info(t); st.code(a); st.success(o)

if __name__ == "__main__": main()

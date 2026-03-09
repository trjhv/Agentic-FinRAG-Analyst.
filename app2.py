import streamlit as st
import pickle
import faiss
import json
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from rank_bm25 import BM25Okapi
import yfinance as yf
import os
import numpy as np
import pandas as pd
import tempfile
import shutil

#st.set_page_config(page_title="Bring Your Own Brain", page_icon="🧠", layout="wide")

# ============================================================================
# LOAD COMPANY LIST FROM CSV
# ============================================================================

@st.cache_data
def load_company_list():
    '''Load company names and tickers from CSV files'''
    try:
        companies = {}
        
        # Try to load both CSV files
        csv_files = ['company_tickers_RAG.csv', 'fortune500_tickers.csv']
        
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    company_name = row['Company'].strip()
                    ticker = row['Ticker'].strip()
                    companies[company_name] = ticker
        
        if not companies:
            # Fallback list if CSV not found
            companies = {
                'Apple': 'AAPL',
                'Microsoft': 'MSFT',
                'Alphabet Inc.': 'GOOGL',
                'Amazon': 'AMZN',
                'Tesla': 'TSLA'
            }
        
        return companies
    except Exception as e:
        st.error(f"Error loading companies: {str(e)}")
        return {'Apple': 'AAPL', 'Microsoft': 'MSFT'}

# ============================================================================
# GROQ CLIENT
# ============================================================================

def get_groq_client():
    try:
        if 'GROQ_API_KEY' in st.secrets:
            api_key = st.secrets['GROQ_API_KEY']
        elif 'groq_api_key' in st.session_state:
            api_key = st.session_state.groq_api_key
        else:
            api_key = st.text_input("Enter your Groq API Key:", type="password", key="api_key_input")
            if api_key:
                st.session_state.groq_api_key = api_key
            else:
                st.warning("⚠️ Please enter your Groq API key")
                st.info("Get your FREE API key from: https://console.groq.com/keys")
                return None
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# ============================================================================
# MODELS
# ============================================================================

@st.cache_resource
def load_models():
    try:
        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        return embedding_model, cross_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# ============================================================================
# FILE LOADING (NO PROCESSING!)
# ============================================================================

def load_uploaded_files(faiss_file, metadata_file, bm25_file):
    '''Load pre-computed files directly into memory'''
    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files to temp directory
        faiss_path = os.path.join(temp_dir, 'index.faiss')
        metadata_path = os.path.join(temp_dir, 'metadata.pkl')
        bm25_path = os.path.join(temp_dir, 'bm25.pkl')
        
        with open(faiss_path, 'wb') as f:
            f.write(faiss_file.read())
        
        with open(metadata_path, 'wb') as f:
            f.write(metadata_file.read())
        
        with open(bm25_path, 'wb') as f:
            f.write(bm25_file.read())
        
        # Load FAISS index
        faiss_index = faiss.read_index(faiss_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Load BM25
        with open(bm25_path, 'rb') as f:
            bm25 = pickle.load(f)
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
        
        return faiss_index, metadata, bm25
    
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None

# ============================================================================
# AGENTIC TOOLS
# ============================================================================

def tool_rag_search(query: str, top_k: int = 5) -> str:
    '''Search the company's 10-K data'''
    try:
        embedding_model = st.session_state.get('embedding_model')
        cross_encoder = st.session_state.get('cross_encoder')
        faiss_index = st.session_state.get('faiss_index')
        metadata = st.session_state.get('metadata')
        bm25 = st.session_state.get('bm25')
        
        if not all([embedding_model, faiss_index, metadata, bm25]):
            return "Error: No company data loaded. Please upload files first."
        
        # Hybrid search
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        
        distances, indices = faiss_index.search(query_embedding, min(top_k * 2, len(metadata)))
        
        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        combined_results = {}
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(metadata) and idx >= 0:
                similarity = 1 / (1 + float(score))
                combined_results[int(idx)] = {'faiss_score': similarity, 'bm25_score': 0}
        
        for idx, score in enumerate(bm25_scores):
            if idx in combined_results:
                combined_results[idx]['bm25_score'] = float(score)
        
        if combined_results:
            max_faiss = max([v['faiss_score'] for v in combined_results.values()]) or 1
            max_bm25 = max([v['bm25_score'] for v in combined_results.values()]) or 1
            
            for idx in combined_results:
                combined_results[idx]['total'] = (
                    0.7 * (combined_results[idx]['faiss_score'] / max_faiss) + 
                    0.3 * (combined_results[idx]['bm25_score'] / max_bm25)
                )
        
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['total'], reverse=True)[:top_k]
        contexts = [metadata[idx] for idx, _ in sorted_results]
        
        # Rerank
        if cross_encoder and contexts:
            pairs = [[query, ctx['text']] for ctx in contexts]
            scores = cross_encoder.predict(pairs)
            ranked_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
            contexts = [ctx for ctx, _ in ranked_contexts[:3]]
        
        company = st.session_state.get('selected_company', 'the company')
        result = f"From {company}'s 10-K:\n\n" + "\n\n".join([f"[Context {i+1}]: {ctx['text'][:500]}..." for i, ctx in enumerate(contexts)])
        return result
    except Exception as e:
        return f"Error in search: {str(e)}"

def tool_get_stock_data(ticker: str = None) -> str:
    '''Get current stock data'''
    try:
        if not ticker:
            ticker = st.session_state.get('company_ticker', 'AAPL')
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            "Symbol": ticker,
            "Company": info.get('longName', ticker),
            "Current Price": f"${info.get('currentPrice', 'N/A')}",
            "Market Cap": f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else 'N/A',
            "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A',
            "52 Week High": f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52 Week Low": f"${info.get('fiftyTwoWeekLow', 'N/A')}"
        }
        
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

def tool_calculate(expression: str) -> str:
    '''Perform calculations'''
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def tool_financial_ratios(metric: str, value1: float, value2: float = None) -> str:
    '''Calculate financial ratios'''
    try:
        metric = metric.lower()
        if metric == "profit_margin" and value2:
            result = (value1 / value2) * 100
            return f"Profit Margin: {result:.2f}%"
        elif metric == "pe_ratio" and value2:
            result = value1 / value2
            return f"P/E Ratio: {result:.2f}"
        elif metric == "roe" and value2:
            result = (value1 / value2) * 100
            return f"ROE: {result:.2f}%"
        else:
            return "Supported: profit_margin, pe_ratio, roe"
    except Exception as e:
        return f"Error: {str(e)}"

def tool_compare_companies(ticker1: str, ticker2: str) -> str:
    '''Compare two companies'''
    try:
        stock1 = yf.Ticker(ticker1)
        stock2 = yf.Ticker(ticker2)
        info1 = stock1.info
        info2 = stock2.info
        
        comparison = {
            ticker1: {
                "Price": f"${info1.get('currentPrice', 'N/A')}",
                "Market Cap": f"${info1.get('marketCap', 0)/1e9:.2f}B" if info1.get('marketCap') else 'N/A',
                "P/E": f"{info1.get('trailingPE', 'N/A'):.2f}" if info1.get('trailingPE') else 'N/A'
            },
            ticker2: {
                "Price": f"${info2.get('currentPrice', 'N/A')}",
                "Market Cap": f"${info2.get('marketCap', 0)/1e9:.2f}B" if info2.get('marketCap') else 'N/A',
                "P/E": f"{info2.get('trailingPE', 'N/A'):.2f}" if info2.get('trailingPE') else 'N/A'
            }
        }
        
        return json.dumps(comparison, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_rag_search",
            "description": "Search the company's uploaded 10-K report data",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_stock_data",
            "description": "Get live stock market data",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_financial_ratios",
            "description": "Calculate financial ratios",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string"},
                    "value1": {"type": "number"},
                    "value2": {"type": "number"}
                },
                "required": ["metric", "value1"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_compare_companies",
            "description": "Compare two companies",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker1": {"type": "string"},
                    "ticker2": {"type": "string"}
                },
                "required": ["ticker1", "ticker2"]
            }
        }
    }
]

# ============================================================================
# REACT AGENT
# ============================================================================

def execute_tool(tool_name: str, tool_args: dict) -> str:
    tool_functions = {
        "tool_rag_search": tool_rag_search,
        "tool_get_stock_data": tool_get_stock_data,
        "tool_calculate": tool_calculate,
        "tool_financial_ratios": tool_financial_ratios,
        "tool_compare_companies": tool_compare_companies
    }
    
    if tool_name in tool_functions:
        try:
            return tool_functions[tool_name](**tool_args)
        except Exception as e:
            return f"Error: {str(e)}"
    return f"Unknown tool: {tool_name}"

def react_agent(query: str, groq_client: Groq, max_iterations: int = 5) -> dict:
    thoughts = []
    actions = []
    observations = []
    
    company = st.session_state.get('selected_company', 'the company')
    ticker = st.session_state.get('company_ticker', '')
    
    messages = [
        {
            "role": "system",
            "content": f'''You are a financial analysis expert analyzing {company} ({ticker}).

Use ReAct pattern: Think → Act (use tools) → Observe → Repeat.

Available tools:
- tool_rag_search: Search the uploaded 10-K data
- tool_get_stock_data: Get live market data
- tool_calculate: Math operations
- tool_financial_ratios: Calculate ratios
- tool_compare_companies: Compare companies

Be accurate. Use tools, don't guess.'''
        },
        {"role": "user", "content": query}
    ]
    
    for iteration in range(max_iterations):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048
            )
            
            assistant_message = response.choices[0].message
            
            if assistant_message.tool_calls:
                thoughts.append(f"💭 Iteration {iteration + 1}: Using tools")
                
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    actions.append(f"🔧 {tool_name}({json.dumps(tool_args)})")
                    tool_result = execute_tool(tool_name, tool_args)
                    observations.append(f"👁️ {tool_result[:300]}...")
                    
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})
            else:
                return {
                    'answer': assistant_message.content,
                    'thoughts': thoughts,
                    'actions': actions,
                    'observations': observations,
                    'iterations': iteration + 1
                }
        except Exception as e:
            return {'answer': f"Error: {str(e)}", 'thoughts': thoughts, 'actions': actions, 'observations': observations, 'iterations': iteration + 1}
    
    return {'answer': "Max iterations reached", 'thoughts': thoughts, 'actions': actions, 'observations': observations, 'iterations': max_iterations}

# ============================================================================
# MAIN UI
# ============================================================================

def main():
    st.title("🧠 Bring Your Own Brain - Multi-Company RAG")
    st.markdown("**Upload Pre-Processed Data | Zero Processing Lag | Full Agentic Analysis** | Powered by Groq LPU ⚡")
    
    # Initialize Groq
    groq_client = get_groq_client()
    if not groq_client:
        return
    
    st.success("✅ Groq LPU Connected!")
    
    # Load models
    embedding_model, cross_encoder = load_models()
    if not all([embedding_model, cross_encoder]):
        st.error("Model loading failed")
        return
    
    st.session_state.embedding_model = embedding_model
    st.session_state.cross_encoder = cross_encoder
    
    # Load company list
    companies = load_company_list()
    
    st.markdown("---")
    st.markdown("### 🏢 Step 1: Select Company")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_company = st.selectbox(
            "Choose Company:",
            options=sorted(companies.keys()),
            index=0
        )
    
    with col2:
        company_ticker = st.text_input(
            "Ticker Symbol:",
            value=companies.get(selected_company, 'AAPL')
        ).upper()
    
    st.session_state.selected_company = selected_company
    st.session_state.company_ticker = company_ticker
    
    # Upload section
    st.markdown("---")
    st.markdown("### 📤 Step 2: Upload Pre-Processed Files")
    
    st.info(f'''
    **Upload 4 files for {selected_company}:**
    1. **10-K PDF** (for reference - not processed)
    2. **FAISS Index** (.faiss file)
    3. **Metadata** (.pkl file) 
    4. **BM25 Index** (.pkl file)
    
    ⚡ **Zero lag** - We load your pre-computed indices directly!
    ''')
    
    col1, col2 = st.columns(2)
    
    with col1:
        pdf_file = st.file_uploader(
            "📄 10-K PDF (Reference)",
            type=['pdf'],
            help="Upload for reference - not processed",
            key="pdf_upload"
        )
        
        faiss_file = st.file_uploader(
            "🔍 FAISS Index (.faiss)",
            type=['faiss'],
            help="Pre-computed FAISS vector index",
            key="faiss_upload"
        )
    
    with col2:
        metadata_file = st.file_uploader(
            "📊 Metadata (.pkl)",
            type=['pkl'],
            help="Pre-computed metadata pickle file",
            key="metadata_upload"
        )
        
        bm25_file = st.file_uploader(
            "📈 BM25 Index (.pkl)",
            type=['pkl'],
            help="Pre-computed BM25 keyword index",
            key="bm25_upload"
        )
    
    # Load button
    if all([faiss_file, metadata_file, bm25_file]):
        if st.button("🚀 Load Data (No Processing!)", type="primary"):
            with st.spinner(f"⚡ Loading {selected_company}'s pre-processed data..."):
                faiss_index, metadata, bm25 = load_uploaded_files(
                    faiss_file, metadata_file, bm25_file
                )
                
                if all([faiss_index, metadata, bm25]):
                    st.session_state.faiss_index = faiss_index
                    st.session_state.metadata = metadata
                    st.session_state.bm25 = bm25
                    
                    if pdf_file:
                        st.session_state.pdf_file = pdf_file
                    
                    st.success(f"✅ Loaded {len(metadata)} pre-computed chunks for {selected_company}!")
                    st.balloons()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Live Market Data")
        if company_ticker:
            try:
                stock = yf.Ticker(company_ticker)
                info = stock.info
                if info.get('currentPrice'):
                    st.metric("Price", f"${info.get('currentPrice', 'N/A')}")
                    st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A')
                    st.metric("P/E", f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else 'N/A')
            except:
                st.info("Unable to fetch live data")
        
        if 'faiss_index' in st.session_state:
            st.markdown("---")
            st.markdown("### ✅ Data Loaded")
            st.metric("Chunks", len(st.session_state.metadata))
            st.success("Ready for queries!")
        
        st.markdown("---")
        st.markdown("### 🛠️ Available Tools")
        st.markdown('''
        - 📊 Live Stock Data
        - 🔍 10-K Search
        - 🧮 Calculator
        - 📈 Financial Ratios
        - ⚖️ Company Comparison
        ''')
    
    # Query interface
    if 'faiss_index' in st.session_state:
        st.markdown("---")
        st.markdown("### 💬 Step 3: Ask Questions")
        
        with st.expander("💡 Example Queries"):
            st.markdown(f'''
            **Multi-Hop Reasoning:**
            - What was {selected_company}'s revenue in the 10-K and what's the current P/E ratio?
            - Compare the profit margin from the 10-K with current stock price
            - What risks are mentioned and how does the stock compare to 52-week high?
            
            **Calculations:**
            - Calculate revenue growth if 2021 revenue is in the 10-K
            - What percentage is R&D spending of current market cap?
            
            **Comparisons:**
            - Compare {company_ticker} with MSFT on key metrics
            ''')
        
        query = st.text_area(
            f"Ask about {selected_company}:",
            placeholder=f"e.g., What was {selected_company}'s revenue in the 10-K and what's the current stock price?",
            height=100,
            key="query_input"
        )
        
        if st.button("🚀 Run Agent", type="primary"):
            if query:
                with st.spinner("🤖 Agent analyzing..."):
                    result = react_agent(query, groq_client, max_iterations=5)
                
                st.markdown("---")
                st.markdown("### 💡 Answer")
                st.markdown(result['answer'])
                
                if result['thoughts'] or result['actions']:
                    st.markdown("---")
                    with st.expander("🧠 View Agent Reasoning", expanded=True):
                        st.markdown(f"**Total Iterations:** {result['iterations']}")
                        
                        for i, (thought, action, obs) in enumerate(zip(
                            result['thoughts'],
                            result['actions'],
                            result['observations']
                        )):
                            st.markdown(f"**Step {i+1}:**")
                            st.info(thought)
                            st.code(action, language="text")
                            st.success(obs)
                            if i < len(result['thoughts']) - 1:
                                st.markdown("---")
    else:
        st.info("👆 Upload your pre-processed files to start analyzing")
        
        st.markdown("---")
        st.markdown("### ⚡ Why This is Fast:")
        st.markdown('''
        **Traditional Approach:** 📄 → 🔪 Chunk → 🧮 Embed → 💾 Index → ⏰ **2-5 minutes**
        
        **Our Approach:** 📤 Upload pre-computed files → ⚡ **Instant!**
        
        **You bring:**
        - Your own FAISS index
        - Your own metadata
        - Your own BM25 index
        - The original PDF (optional)
        
        **We provide:**
        - Blazing fast Groq LPU
        - Agentic ReAct reasoning
        - Multi-hop queries
        - Live stock data integration
        ''')

if __name__ == "__main__":
    main()

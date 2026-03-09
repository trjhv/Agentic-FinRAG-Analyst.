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
import re
from datetime import datetime
import requests

#st.set_page_config(page_title="Agentic RAG System", page_icon="🤖", layout="wide")

# ============================================================================
# GROQ CLIENT SETUP
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
        st.error(f"Error setting up Groq client: {str(e)}")
        return None

# ============================================================================
# MODEL LOADING
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

def detect_file_format():
    langchain_files = ['index.faiss', 'apple_documents.pkl']
    langchain_exists = all(os.path.exists(f) for f in langchain_files)
    if langchain_exists:
        return 'langchain'
    elif os.path.exists('index (1).faiss'):
        return 'langchain_alt'
    return None

@st.cache_data
def load_langchain_format():
    try:
        if os.path.exists('index.faiss'):
            faiss_index = faiss.read_index('index.faiss')
        elif os.path.exists('index (1).faiss'):
            faiss_index = faiss.read_index('index (1).faiss')
        else:
            return None, None, None
        
        if os.path.exists('apple_documents.pkl'):
            with open('apple_documents.pkl', 'rb') as f:
                documents = pickle.load(f)
        else:
            return None, None, None
        
        metadata = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                text = doc.page_content
                meta = doc.metadata if hasattr(doc, 'metadata') else {}
            else:
                text = str(doc)
                meta = {}
            metadata.append({'text': text, 'metadata': meta})
        
        tokenized_docs = [doc['text'].lower().split() for doc in metadata]
        bm25 = BM25Okapi(tokenized_docs)
        return faiss_index, metadata, bm25
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_data
def load_apple_data():
    file_format = detect_file_format()
    if file_format in ['langchain', 'langchain_alt']:
        st.info("📁 Loading Apple 10-K data...")
        return load_langchain_format()
    else:
        st.error("❌ No data files found!")
        return None, None, None

# ============================================================================
# AGENTIC TOOLS - THE BRAIN OF THE SYSTEM
# ============================================================================

def tool_rag_search(query: str, top_k: int = 5) -> str:
    """Search Apple's 10-K report for relevant information"""
    try:
        embedding_model = st.session_state.get('embedding_model')
        cross_encoder = st.session_state.get('cross_encoder')
        faiss_index = st.session_state.get('faiss_index')
        metadata = st.session_state.get('metadata')
        bm25 = st.session_state.get('bm25')
        
        if not all([embedding_model, faiss_index, metadata, bm25]):
            return "Error: RAG system not initialized"
        
        # Hybrid search
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
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
        
        result = "\n\n".join([f"[Context {i+1}]: {ctx['text'][:500]}..." for i, ctx in enumerate(contexts)])
        return result
    except Exception as e:
        return f"Error in RAG search: {str(e)}"

def tool_get_stock_data(ticker: str = "AAPL") -> str:
    """Get current stock data including price, P/E ratio, market cap, etc."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        data = {
            "Symbol": ticker,
            "Current Price": f"${info.get('currentPrice', 'N/A')}",
            "Market Cap": f"${info.get('marketCap', 0)/1e9:.2f}B",
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "Forward P/E": info.get('forwardPE', 'N/A'),
            "Dividend Yield": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A',
            "52 Week High": f"${info.get('fiftyTwoWeekHigh', 'N/A')}",
            "52 Week Low": f"${info.get('fiftyTwoWeekLow', 'N/A')}",
            "Volume": info.get('volume', 'N/A'),
            "Avg Volume": info.get('averageVolume', 'N/A')
        }
        
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error fetching stock data: {str(e)}"

def tool_calculate(expression: str) -> str:
    """Perform mathematical calculations. Use Python syntax."""
    try:
        # Safe eval with limited namespace
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "len": len
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

def tool_financial_ratios(metric: str, value1: float, value2: float = None) -> str:
    """Calculate common financial ratios.
    Supported metrics: pe_ratio, profit_margin, roe, debt_to_equity, current_ratio
    """
    try:
        metric = metric.lower()
        
        if metric == "pe_ratio" and value2:
            # P/E = Market Price / Earnings Per Share
            result = value1 / value2
            return f"P/E Ratio: {result:.2f}"
        
        elif metric == "profit_margin" and value2:
            # Profit Margin = (Net Income / Revenue) * 100
            result = (value1 / value2) * 100
            return f"Profit Margin: {result:.2f}%"
        
        elif metric == "roe" and value2:
            # ROE = (Net Income / Shareholder Equity) * 100
            result = (value1 / value2) * 100
            return f"Return on Equity: {result:.2f}%"
        
        elif metric == "debt_to_equity" and value2:
            # Debt-to-Equity = Total Debt / Shareholder Equity
            result = value1 / value2
            return f"Debt-to-Equity Ratio: {result:.2f}"
        
        elif metric == "current_ratio" and value2:
            # Current Ratio = Current Assets / Current Liabilities
            result = value1 / value2
            return f"Current Ratio: {result:.2f}"
        
        else:
            return f"Unknown metric: {metric}. Supported: pe_ratio, profit_margin, roe, debt_to_equity, current_ratio"
    except Exception as e:
        return f"Error calculating ratio: {str(e)}"

def tool_web_search(query: str) -> str:
    """Search the web for current information (simulated - replace with real API)"""
    # Note: This is a placeholder. For production, use a real search API like Serper, Tavily, etc.
    return f"Web search for '{query}' - Note: Web search not yet implemented. Use RAG search for 10-K data or stock data for live prices."

# ============================================================================
# TOOL DEFINITIONS FOR GROQ
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "tool_rag_search",
            "description": "Search Apple's 10-K financial report for information about revenue, expenses, risks, products, strategy, R&D spending, and other company information from the official filing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information in the 10-K report"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_get_stock_data",
            "description": "Get current live stock market data including current price, P/E ratio, market cap, dividend yield, 52-week high/low, and trading volume.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (default: AAPL for Apple)",
                        "default": "AAPL"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_calculate",
            "description": "Perform mathematical calculations. Use for adding, subtracting, multiplying, dividing numbers or computing percentages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression in Python syntax (e.g., '100 * 1.5', '(365.8 / 274.5) * 100')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_financial_ratios",
            "description": "Calculate common financial ratios like P/E ratio, profit margin, ROE, debt-to-equity, current ratio.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Ratio to calculate: pe_ratio, profit_margin, roe, debt_to_equity, current_ratio"
                    },
                    "value1": {
                        "type": "number",
                        "description": "First value (numerator)"
                    },
                    "value2": {
                        "type": "number",
                        "description": "Second value (denominator)"
                    }
                },
                "required": ["metric", "value1"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tool_web_search",
            "description": "Search the web for current information not available in the 10-K report.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ============================================================================
# REACT AGENT - THE CORE REASONING LOOP
# ============================================================================

def execute_tool(tool_name: str, tool_args: dict) -> str:
    """Execute a tool and return its result"""
    tool_functions = {
        "tool_rag_search": tool_rag_search,
        "tool_get_stock_data": tool_get_stock_data,
        "tool_calculate": tool_calculate,
        "tool_financial_ratios": tool_financial_ratios,
        "tool_web_search": tool_web_search
    }
    
    if tool_name in tool_functions:
        try:
            result = tool_functions[tool_name](**tool_args)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    else:
        return f"Unknown tool: {tool_name}"

def react_agent(query: str, groq_client: Groq, max_iterations: int = 5) -> dict:
    """
    ReAct Agent Loop: Reason + Act
    Returns: {
        'answer': final answer,
        'thoughts': list of reasoning steps,
        'actions': list of tool calls,
        'observations': list of tool results
    }
    """
    thoughts = []
    actions = []
    observations = []
    
    messages = [
        {
            "role": "system",
            "content": """You are a financial analysis expert with access to Apple's 10-K report and live market data.

Use the ReAct pattern:
1. THOUGHT: Analyze what information you need
2. ACTION: Use tools to gather information
3. OBSERVATION: Review tool results
4. Repeat until you can answer

Available tools:
- tool_rag_search: Search Apple's 10-K report
- tool_get_stock_data: Get live stock prices and metrics
- tool_calculate: Perform math calculations
- tool_financial_ratios: Calculate financial ratios
- tool_web_search: Search the web

Always use tools to get accurate data. Don't make up numbers."""
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    for iteration in range(max_iterations):
        try:
            # Call Groq with tools
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048
            )
            
            assistant_message = response.choices[0].message
            
            # Check if agent wants to use tools
            if assistant_message.tool_calls:
                # Agent decided to use tools
                thoughts.append(f"💭 Iteration {iteration + 1}: Need to use tools")
                
                # Add assistant message to conversation
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    actions.append(f"🔧 {tool_name}({json.dumps(tool_args)})")
                    
                    # Execute tool
                    tool_result = execute_tool(tool_name, tool_args)
                    observations.append(f"👁️ {tool_result[:300]}...")
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
            else:
                # Agent has final answer
                final_answer = assistant_message.content
                thoughts.append(f"💭 Final answer formulated")
                
                return {
                    'answer': final_answer,
                    'thoughts': thoughts,
                    'actions': actions,
                    'observations': observations,
                    'iterations': iteration + 1
                }
        
        except Exception as e:
            return {
                'answer': f"Error in ReAct loop: {str(e)}",
                'thoughts': thoughts,
                'actions': actions,
                'observations': observations,
                'iterations': iteration + 1
            }
    
    # Max iterations reached
    return {
        'answer': "Maximum iterations reached. Please try a simpler query.",
        'thoughts': thoughts,
        'actions': actions,
        'observations': observations,
        'iterations': max_iterations
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("🤖 Agentic Financial RAG System")
    st.markdown("**ReAct Agent** with Multi-Hop Reasoning | Powered by Groq LPU ⚡")
    
    # Initialize Groq
    groq_client = get_groq_client()
    if not groq_client:
        return
    
    st.success("✅ Groq LPU Connected!")
    
    # Load models and data
    with st.spinner("Loading AI models and data..."):
        embedding_model, cross_encoder = load_models()
        faiss_index, metadata, bm25 = load_apple_data()
    
    if not all([embedding_model, cross_encoder, faiss_index, metadata, bm25]):
        st.error("Failed to load required models or data.")
        return
    
    # Store in session state for tools
    st.session_state.embedding_model = embedding_model
    st.session_state.cross_encoder = cross_encoder
    st.session_state.faiss_index = faiss_index
    st.session_state.metadata = metadata
    st.session_state.bm25 = bm25
    
    st.success(f"✅ Loaded {len(metadata)} document chunks | 🤖 Agentic Mode Active")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Available Tools")
        st.markdown("""
        The agent can use:
        - 📊 **Stock Data** (live prices)
        - 🔍 **RAG Search** (10-K report)
        - 🧮 **Calculator** (math)
        - 📈 **Financial Ratios** (metrics)
        - 🌐 **Web Search** (coming soon)
        """)
        
        st.markdown("---")
        st.header("📊 Live Market Data")
        try:
            stock = yf.Ticker("AAPL")
            info = stock.info
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.2f}B")
            st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
        except:
            st.info("Unable to fetch live data")
        
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        examples = [
            "What was Apple's revenue in 2021 and how does it compare to 2020?",
            "What's Apple's current P/E ratio and is it mentioned in the 10-K?",
            "Calculate Apple's profit margin if revenue is $365.8B and net income is $94.7B",
            "What risks does Apple face according to the 10-K?",
            "Compare Apple's current stock price to its 52-week high"
        ]
        
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:30]}", use_container_width=True):
                st.session_state.query = ex
    
    # Main interface
    st.markdown("### 💬 Ask the Agentic System")
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., What was Apple's net income in 2021 and what's the current P/E ratio?",
        height=100
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🚀 Run Agent", type="primary")
    
    if search_button and query:
        with st.spinner("🤖 Agent thinking and acting..."):
            result = react_agent(query, groq_client, max_iterations=5)
        
        # Display results
        st.markdown("---")
        st.markdown("### 💡 Final Answer")
        st.markdown(result['answer'])
        
        # Show reasoning process
        if result['thoughts'] or result['actions']:
            st.markdown("---")
            st.markdown("### 🧠 Agent Reasoning Process")
            
            with st.expander("🔍 View Reasoning Steps", expanded=True):
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
                    st.markdown("---")

if __name__ == "__main__":
    main()

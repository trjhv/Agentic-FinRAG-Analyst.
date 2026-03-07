import streamlit as st
import sys
import importlib

st.set_page_config(
    page_title="Financial Document Analysis RAG System",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown(
    '''
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        text-align: center;
    }
    .stMarkdown p {
        text-align: center;
        font-size: 18px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Session state for page tracking
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigation function
def navigate_to(page):
    st.session_state.page = page
    if page == 'home':
        for key in list(st.session_state.keys()):
            if key != 'page':
                del st.session_state[key]

# Home Page
def home_page():
    st.title("Welcome to the Financial Document Analysis RAG System 💰")
    
    st.markdown('''
    ### About This Application
    This **financial analysis tool** provides two powerful functionalities to help you analyze and query financial documents effectively:
    
    1. **Agentic RAG System** - Query pre-processed financial knowledge base using advanced Retrieval Augmented Generation (RAG) with full ReAct reasoning.
    2. **Multi-Company Vector Upload** - Upload your own pre-computed FAISS indices and BM25 data for any company. Zero processing lag, instant analysis.
    
    Select an option below to get started:
    ''')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Button 1: RAG System
        if st.button("🔍 Agentic RAG System", use_container_width=True, key="rag_btn"):
            navigate_to('rag')
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Button 2: Multi-Company Vector Upload (RENAMED!)
        if st.button("🧠 Multi-Company Vector Upload", use_container_width=True, key="doc_btn"):
            navigate_to('doc_qa')
            st.rerun()
    
    st.markdown("---")
    st.info('''
    ### Why Use This Tool?
    - **Efficient Financial Document Analysis**: Quickly extract insights from large financial documents
    - **Advanced AI-Powered Search**: Combines semantic and keyword search with ReAct reasoning
    - **Agentic Multi-Hop**: Autonomous tool selection and multi-step reasoning
    - **Bring Your Own Brain**: Upload pre-processed indices for instant querying
    - **Zero Processing Lag**: No chunking, no embedding - just upload and analyze
    ''')
    
    st.markdown("---")
    st.markdown('''
    <div style='text-align: center'>
        <p style='color: #666;'>Built with Streamlit, LangChain, Groq LPU, and FAISS</p>
    </div>
    ''', unsafe_allow_html=True)

# Main app logic
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'rag':
    if st.button("← Back to Home", key="back_rag"):
        navigate_to('home')
        st.rerun()
    
    try:
        import App1
        App1.main()
    except Exception as e:
        st.error(f"Error loading Agentic RAG System: {str(e)}")
        st.exception(e)
        st.info("Make sure App1.py is in the same directory")
    
elif st.session_state.page == 'doc_qa':
    if st.button("← Back to Home", key="back_doc"):
        navigate_to('home')
        st.rerun()
    
    try:
        import App2
        App2.main()
    except Exception as e:
        st.error(f"Error loading Multi-Company Vector Upload: {str(e)}")
        st.exception(e)
        st.info("Make sure App2.py is in the same directory")
        st.markdown("**Debug Info:**")
        st.code(str(e))

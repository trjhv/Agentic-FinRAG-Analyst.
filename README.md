# 📈 Agentic FinRAG Analyst

> **Agentic FinRAG Analyst** represents the next evolution in open-source financial intelligence. By moving beyond passive Retrieval-Augmented Generation (RAG), this Streamlit-based application utilizes Llama 3.3 (via Groq LPUs) as an active controller within a ReAct loop. It seamlessly executes multi-hop reasoning by orchestrating hybrid semantic and lexical searches (FAISS + BM25) across static 10-K financial documents, and immediately cross-validating those insights against real-time market data via the `yfinance` API. Engineered specifically for absolute resilience on constrained free-tier cloud environments (1GB RAM), the system introduces deterministic intent routing to minimize LLM tool-calling overhead, dynamic context truncation for strict token optimization, and custom exponential backoff algorithms to gracefully handle API rate limits. The result is an institutional-grade, zero-hallucination analysis engine built entirely on accessible, open-weight architecture.



## 🛡️ Strategic Advantages & Defense Logic

* **Superior to Closed Models (e.g., Claude/GPT-4 for Finance):** Standard LLMs suffer from temporal hallucinations because their knowledge is frozen in time. This application mitigates that entirely by grounding reasoning in **real-time API access**, ensuring 0-hallucination execution on current market conditions.
* **Agentic (Active) vs. Standard FinRAG (Passive):** Unlike standard GitHub FinRAG repositories that merely retrieve and summarize text, this system is an active agent. It dynamically decides *when* to search a PDF, *when* to pull live ticker data, and *how* to synthesize both to answer complex user queries.

---

## 🏗️ Technical Stack

| Component | Technology | Implementation Details |
| :--- | :--- | :--- |
| **LLM Controller** | Llama 3.3 70B | Hosted via Groq API for ultra-fast LPU acceleration and rapid ReAct iterations. |
| **Embeddings** | `all-MiniLM-L6-v2` | Lightweight embedding model optimized specifically for Streamlit Cloud's 1GB RAM limit. |
| **Vector Store** | FAISS + BM25 | Hybrid Search: FAISS for dense semantic meaning, BM25 for sparse lexical/keyword accuracy (crucial for exact ticker symbols and financial jargon). |
| **Live Data API** | `yfinance` | Executes real-time stock price and market data retrieval. |
| **Frontend UI** | Streamlit | Multi-Page Application routing for modular financial analysis. |

---

## 🧠 Agentic Logic & Workflow

1. **Deterministic Tool Routing:** Pre-evaluates user intent based on queries to route deterministically, saving unnecessary LLM thought generation.
2. **ReAct Reasoning Loop:** The agent uses a structured **Reason + Act** paradigm to break down complex financial queries, execute intermediate searches, and evaluate the results before generating a final answer.
3. **Multi-Hop Synthesis:** Capable of bridging disparate data sources. For example, it can extract historical static data (like a P/E ratio from a 10-K PDF) and combine it with live execution data (current stock price) to calculate adjusted, real-time valuations.

---

## ⚙️ Free-Tier Resilience Strategy

Building production-ready AI on free-tier infrastructure requires strict defensive engineering. This application implements a three-pillar resilience strategy:

1. **Safe Retry Logic:** Custom `exponential backoff` algorithms specifically handle Groq API `429 Rate Limit` errors, ensuring the app degrades gracefully rather than crashing during peak loads.
2. **Keyword Routing:** Implements lightweight pre-checks for specific "market" keywords. This bypasses the heavy ReAct tool-calling loop when a user simply wants a stock price, saving exactly 1 LLM API call per turn.
3. **Strict Context Truncation:** The hybrid retriever is hard-coded to feed only the top-3 most relevant document chunks into the prompt, strictly maintaining token quotas and preventing context-window overflows.

---

## 📂 Project Structure

```text
agentic-finrag-analyst/
├── main.py              # Streamlit Multi-Page Entry Point
├── app1.py              # Page 1: Static Document Intelligence (10-K PDFs)
├── app2.py              # Page 2: Real-Time Market Agent
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation

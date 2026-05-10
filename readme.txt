The Auditor : AI Hedge Fund Research Assistant
This tool analyzes 10-K and 10-Q SEC filings from local files and generates structured investment research notes using Gemini AI.

Architecture :
data/filings/<TICKER>/<filing>.txt  (or .pdf)
         ↓
    loader.py           ← reads txt/pdf
         ↓
    chunker.py          ← splits into 1500-char overlapping chunks
         ↓
    vectorstore.py      ← embeds with sentence-transformers → ChromaDB
         ↓
    report_generator.py ← retrieves relevant chunks → Gemini → JSON report
         ↓
    app.py              ← Streamlit UI renders the report

Features :
1. Local file support (.txt and .pdf)
2. Multi-company support (AAPL, TSLA, MSFT, GOOGL, AMZN + any ticker)
3. RAG pipeline (ChromaDB + sentence-transformers, fully local)
4. Structured research notes (Bull/Bear/Risks/Score/Recommendation)
5. Live financial metrics from Yahoo Finance
6. Focus area selection (Revenue, Risks, Cash Flow, etc.)
7.Persist ChromaDB to disk(avoid re-embedding on every run)

to add:
1. Conversational follow-up Q&A
2. add competitor side by side analysis
3. instead of using local files use SEC-EDGAR api call to fetch files
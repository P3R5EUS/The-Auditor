import streamlit as st
import os
from pathlib import Path
from pipeline.loader import load_filing
from pipeline.chunker import chunk_text
from pipeline.vectorstore import build_vectorstore, query_vectorstore
from pipeline.report_generator import generate_report
from pipeline.financials import get_financial_snapshot
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Hedge Fund Research Assistant",
    page_icon="📊",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Serif+Display&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0a0e17;
    color: #e2e8f0;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #f1f5f9 !important;
}

.ticker-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1e3a5f, #0f2744);
    border: 1px solid #2d6a9f;
    color: #60a5fa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

.score-ring {
    text-align: center;
    padding: 1.5rem;
    background: #0f1929;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
}

.score-number {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.5rem;
    font-weight: 600;
    line-height: 1;
}

.section-card {
    background: #0f1929;
    border: 1px solid #1e2d45;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.bull { border-left: 3px solid #22c55e; }
.bear { border-left: 3px solid #ef4444; }
.risk { border-left: 3px solid #f59e0b; }
.overview { border-left: 3px solid #60a5fa; }

.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.6rem;
}

.tag-bull { background: #14532d; color: #4ade80; }
.tag-bear { background: #450a0a; color: #f87171; }
.tag-risk { background: #431407; color: #fbbf24; }
.tag-overview { background: #0c2a4a; color: #93c5fd; }

.metric-box {
    background: #0f1929;
    border: 1px solid #1e2d45;
    border-radius: 8px;
    padding: 0.9rem 1rem;
    text-align: center;
}

.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e2e8f0;
}

.stSelectbox label, .stTextInput label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-family: 'IBM Plex Mono', monospace !important;
}

div[data-testid="stSelectbox"] > div {
    background: #0f1929 !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    transform: translateY(-1px);
}

.sidebar-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2d45;
}

.filing-pill {
    background: #0f1929;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #60a5fa;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.recommendation-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Research Assistant")
    st.markdown('<div class="sidebar-header">Configuration</div>', unsafe_allow_html=True)

    DATA_DIR = Path("data/filings")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Detect available companies
    available = {}
    for folder in sorted(DATA_DIR.iterdir()):
        if folder.is_dir():
            files = list(folder.glob("*.txt")) + list(folder.glob("*.pdf"))
            if files:
                available[folder.name.upper()] = files

    if not available:
        st.warning("No filings found in `data/filings/`")
        st.markdown("""
        **Setup:**
        ```
        data/filings/
        ├── AAPL/
        │   ├── AAPLQ1.pdf
        │   ├── AAPLQ2.pdf
        │   ├── AAPLQ3.pdf
        │   ├── AAPLQ4.pdf
        │   └── AAPLK.pdf
        ├── MSFT/
        │   ├── MSFTQ1.pdf
        │   ├── MSFTQ2.pdf
        │   ├── MSFTQ3.pdf
        │   ├── MSFTQ4.pdf
        │   └── MSFTK.pdf
        └── NVDA/
            ├── NVDAQ1.pdf
            ├── NVDAQ2.pdf
            ├── NVDAQ3.pdf
            └── NVDAK.pdf
        ```
        Place your downloaded 10-K `.txt` or `.pdf` files in the matching ticker folder.
        """)
        st.stop()

    ticker = st.selectbox(
        "Select Company",
        options=list(available.keys()),
        index=0
    )

    filing_paths = available[ticker]

    st.markdown('<div class="sidebar-header" style="margin-top:1.5rem;">Loaded Filings</div>', unsafe_allow_html=True)
    for fpath in filing_paths:
        size_kb = fpath.stat().st_size // 1024
        st.markdown(
            f'<div class="filing-pill">📄 {fpath.name} <span style="color:#475569;margin-left:auto">{size_kb}KB</span></div>',
            unsafe_allow_html=True)

    st.divider()
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.markdown('<div class="sidebar-header" style="margin-top:1rem;">Analysis Focus</div>', unsafe_allow_html=True)
    focus_areas = st.multiselect(
        "Deep-dive sections",
        ["Revenue & Growth", "Risk Factors", "Competition", "Management Discussion", "Cash Flow", "Debt & Leverage"],
        default=["Revenue & Growth", "Risk Factors"]
    )

    analyze_btn = st.button("⚡ Generate Research Note")


# ── Main Area ─────────────────────────────────────────────────────────────────
st.markdown(f'<div class="ticker-badge">{ticker}</div>', unsafe_allow_html=True)
st.markdown(f"### Annual Report Analysis — {len(filing_paths)} filing(s) loaded")
st.markdown('<hr style="border-color:#1e2d45;margin:0.5rem 0 1.5rem 0">', unsafe_allow_html=True)

if not analyze_btn:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;color:#334155;">
        <div style="font-size:3rem;margin-bottom:1rem;">📑</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;letter-spacing:0.1em;text-transform:uppercase;">
            Configure your analysis in the sidebar and click Generate
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not os.environ.get("GEMINI_API_KEY"):
    st.error("Please enter your Gemini API key in the sidebar.")
    st.stop()

# ── Run Pipeline ───────────────────────────────────────────────────────────────
from pipeline.vectorstore import is_cached

# ── Step 1: Load ALL files ────────────────────────────────────────────────────
st.markdown("#### ⚙️ Pipeline Progress")
step1_label = st.empty()
step1_bar   = st.progress(0)

step1_label.markdown("**Step 1 / 4 — Reading all filings...**")
all_text = ""
for i, fpath in enumerate(filing_paths):
    step1_bar.progress(int((i / len(filing_paths)) * 100), text=f"Reading {fpath.name}…")
    all_text += f"\n\n=== FILE: {fpath.name} ===\n\n"
    all_text += load_filing(fpath)

step1_bar.progress(100, text=f"✅ Loaded {len(filing_paths)} files — {len(all_text):,} chars")
step1_label.markdown(f"**Step 1 / 4 — Reading PDFs** `{len(filing_paths)} files, {len(all_text):,} chars`")

# ── Step 2: Chunk ─────────────────────────────────────────────────────────────
step2_label = st.empty()
step2_bar   = st.progress(0)

step2_label.markdown("**Step 2 / 4 — Chunking text...**")
step2_bar.progress(20, text="Splitting into overlapping chunks…")
chunks = chunk_text(all_text)
step2_bar.progress(100, text=f"✅ {len(chunks)} chunks ready")
step2_label.markdown(f"**Step 2 / 4 — Chunking** `{len(chunks)} chunks @ 1500 chars each`")

# ── Step 3: Embed → ChromaDB (with live progress, skipped if cached) ──────────
step3_label = st.empty()
step3_bar   = st.progress(0)

cached = is_cached(chunks)
if cached:
    step3_label.markdown("**Step 3 / 4 — Vector Index** `⚡ Loaded from disk cache — skipping embedding`")
    step3_bar.progress(100, text="✅ Cache hit — no re-embedding needed")
    vs = build_vectorstore(chunks)
else:
    step3_label.markdown("**Step 3 / 4 — Embedding chunks into ChromaDB...**")

    def update_embed_progress(frac: float, msg: str):
        pct = int(frac * 100)
        step3_bar.progress(pct, text=msg)

    vs = build_vectorstore(chunks, on_progress=update_embed_progress)
    step3_label.markdown(f"**Step 3 / 4 — Embedding** `{len(chunks)} chunks saved to disk`")

# ── Step 4: Retrieve + Generate ───────────────────────────────────────────────
step4_label = st.empty()
step4_bar   = st.progress(0)
step4_label.markdown("**Step 4 / 4 — Retrieving context sections...**")

# Focus-driven retrieval queries
BASE_QUERIES = {
    "overview": "company overview business segments products services revenue streams",
    "bull":     "growth opportunities revenue increase market expansion competitive advantage future outlook",
    "bear":     "declining revenue challenges competition pressure earnings miss guidance cut",
    "risks":    "risk factors regulatory legal financial operational macroeconomic risks",
}
FOCUS_QUERIES = {
    "Revenue & Growth":      {"bull":     "revenue growth rate YoY quarterly earnings beat guidance raised"},
    "Risk Factors":          {"risks":    "risk factors litigation regulatory fines penalties sanctions"},
    "Competition":           {"bear":     "competition market share loss competitor pricing pressure"},
    "Management Discussion": {"overview": "management discussion outlook forward guidance CEO commentary"},
    "Cash Flow":             {"bull":     "free cash flow operating cash flow capex buybacks dividends"},
    "Debt & Leverage":       {"risks":    "debt obligations interest expense leverage ratio covenant default"},
}
queries = dict(BASE_QUERIES)
for area in focus_areas:
    if area in FOCUS_QUERIES:
        for key, extra in FOCUS_QUERIES[area].items():
            queries[key] = queries[key] + " " + extra

contexts = {}
query_keys = list(queries.keys())
for i, (key, q) in enumerate(queries.items()):
    step4_bar.progress(int((i / len(query_keys)) * 50), text=f"Retrieving '{key}' section…")
    contexts[key] = query_vectorstore(vs, q, k=6)

step4_bar.progress(50, text="Calling Gemini to generate report…")
step4_label.markdown("**Step 4 / 4 — Generating research note with Gemini...**")

report = generate_report(ticker, contexts, focus_areas)

step4_bar.progress(90, text="Fetching live financials from Yahoo Finance…")
fin = get_financial_snapshot(ticker)

step4_bar.progress(100, text="✅ Done!")
step4_label.markdown("**Step 4 / 4 — Complete** `Report generated`")

st.markdown("---")
# ── Render Report ──────────────────────────────────────────────────────────────

# Row 1: Score + Recommendation + Metrics
col_score, col_rec, col_metrics = st.columns([1, 1, 2])

with col_score:
    score = report.get("sentiment_score", 5)
    color = "#22c55e" if score >= 7 else "#f59e0b" if score >= 5 else "#ef4444"
    st.markdown(f"""
    <div class="score-ring">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:0.5rem;">Sentiment Score</div>
        <div class="score-number" style="color:{color}">{score}<span style="font-size:1.5rem;color:#475569">/10</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_rec:
    rec = report.get("recommendation", "HOLD")
    rec_colors = {"BUY": ("#14532d", "#4ade80"), "HOLD": ("#431407", "#fbbf24"), "SELL": ("#450a0a", "#f87171")}
    bg, fg = rec_colors.get(rec, ("#1e2d45", "#94a3b8"))
    st.markdown(f"""
    <div class="score-ring">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#64748b;margin-bottom:0.5rem;">Recommendation</div>
        <div style="background:{bg};color:{fg};padding:0.5rem 1rem;border-radius:6px;font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:1.6rem;letter-spacing:0.08em;">{rec}</div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    if fin:
        mc, pe, rev, eps = st.columns(4)
        pairs = [
            (mc, "Mkt Cap", fin.get("market_cap", "N/A")),
            (pe, "P/E Ratio", fin.get("pe_ratio", "N/A")),
            (rev, "Revenue", fin.get("revenue", "N/A")),
            (eps, "EPS", fin.get("eps", "N/A")),
        ]
        for col, label, val in pairs:
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{val}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Row 2: Overview
st.markdown(f"""
<div class="section-card overview">
    <span class="tag tag-overview">Overview</span>
    <p style="color:#cbd5e1;line-height:1.75;margin:0">{report.get("overview","")}</p>
</div>
""", unsafe_allow_html=True)

# Row 3: Bull + Bear side by side
col_bull, col_bear = st.columns(2)

with col_bull:
    bull_points = report.get("bull_thesis", [])
    points_html = "".join([f'<li style="margin-bottom:0.6rem;color:#cbd5e1">{p}</li>' for p in bull_points])
    st.markdown(f"""
    <div class="section-card bull">
        <span class="tag tag-bull">🟢 Bull Thesis</span>
        <ul style="margin:0;padding-left:1.2rem;line-height:1.7">{points_html}</ul>
    </div>
    """, unsafe_allow_html=True)

with col_bear:
    bear_points = report.get("bear_thesis", [])
    points_html = "".join([f'<li style="margin-bottom:0.6rem;color:#cbd5e1">{p}</li>' for p in bear_points])
    st.markdown(f"""
    <div class="section-card bear">
        <span class="tag tag-bear">🔴 Bear Thesis</span>
        <ul style="margin:0;padding-left:1.2rem;line-height:1.7">{points_html}</ul>
    </div>
    """, unsafe_allow_html=True)

# Row 4: Key Risks
risk_points = report.get("key_risks", [])
risks_html = "".join([f'<li style="margin-bottom:0.6rem;color:#cbd5e1">{p}</li>' for p in risk_points])
st.markdown(f"""
<div class="section-card risk">
    <span class="tag tag-risk">⚠️ Key Risks</span>
    <ul style="margin:0;padding-left:1.2rem;line-height:1.7">{risks_html}</ul>
</div>
""", unsafe_allow_html=True)

# Row 5: Analyst Notes
if report.get("analyst_notes"):
    st.markdown(f"""
    <div class="section-card" style="border-left:3px solid #7c3aed">
        <span class="tag" style="background:#2e1065;color:#c4b5fd">📝 Analyst Notes</span>
        <p style="color:#cbd5e1;line-height:1.75;margin:0">{report.get("analyst_notes","")}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Follow-up Q&A ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 💬 Ask Follow-up Questions")
st.markdown('<p style="color:#64748b;font-size:0.85rem">Dig deeper into specific sections of the filing.</p>', unsafe_allow_html=True)

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Display history
for qa in st.session_state.qa_history:
    with st.chat_message("user"):
        st.write(qa["q"])
    with st.chat_message("assistant"):
        st.write(qa["a"])

user_q = st.chat_input("e.g. What does the company say about AI competition?")
if user_q:
    with st.spinner("Searching filing..."):
        from pipeline.qa import answer_question
        ctx = query_vectorstore(vs, user_q, k=8)
        answer = answer_question(ticker, user_q, ctx, st.session_state.qa_history)
    st.session_state.qa_history.append({"q": user_q, "a": answer})
    with st.chat_message("user"):
        st.write(user_q)
    with st.chat_message("assistant"):
        st.write(answer)
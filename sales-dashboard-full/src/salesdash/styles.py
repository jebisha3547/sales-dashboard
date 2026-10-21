"""Custom CSS for the Streamlit dashboard (cards, sidebar, tabs, tables, mobile layout).

Colours come from Streamlit's own theme variables (set in .streamlit/config.toml), so the
palette lives in one place. Only CSS is changed here; no data or logic.
"""

CSS = """
/* ---------- page layout ---------- */
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* ---------- page title ---------- */
.page-header h1 { font-size: 1.85rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; padding: 0; }
.page-header p  { margin: 0.2rem 0 1.1rem 0; opacity: 0.65; font-size: 0.95rem; }

/* ---------- KPI / metric cards ---------- */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.22);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 4px 14px rgba(16, 24, 40, 0.10); transform: translateY(-1px); }
[data-testid="stMetricLabel"] { opacity: 0.7; font-size: 0.85rem; }
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    font-size: clamp(1.25rem, 2.1vw, 1.8rem);
    font-weight: 700;
    line-height: 1.2;
    white-space: normal !important;   /* long values wrap onto a second line ... */
    overflow-wrap: anywhere;          /* ... and never spill into the neighbouring card */
}

/* ---------- bordered cards that hold charts and tables ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { border-right: 1px solid rgba(128, 128, 128, 0.22); }
[data-testid="stSidebar"] h1 { font-size: 1.35rem; font-weight: 700; margin-bottom: 0; }
[data-testid="stSidebar"] h2 {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; opacity: 0.6; margin-top: 0.6rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: 0.4rem 0.7rem; border-radius: 8px; margin-bottom: 0.1rem; transition: background 0.15s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(0, 114, 178, 0.08); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(0, 114, 178, 0.12); font-weight: 600;
}

/* ---------- buttons, inputs, tabs ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px; font-weight: 600;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    box-shadow: 0 3px 10px rgba(16, 24, 40, 0.15); transform: translateY(-1px);
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] {
    border-radius: 8px; border: 1px solid rgba(128, 128, 128, 0.28);
}
button[data-baseweb="tab"] { font-weight: 600; padding: 0.5rem 1rem; }

/* ---------- tables, alerts, expanders ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 10px; overflow: hidden;
}
[data-testid="stAlert"], [data-testid="stExpander"] { border-radius: 10px; }

/* ---------- small screens ---------- */
@media (max-width: 640px) {
    .block-container { padding: 1rem 0.8rem 2rem 0.8rem; }
    .page-header h1 { font-size: 1.45rem; }
    [data-testid="stMetric"] { padding: 0.6rem 0.8rem; }
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] > div { font-size: 1.3rem; }
    button[data-baseweb="tab"] { padding: 0.4rem 0.6rem; }
}
"""
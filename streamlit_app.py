import os, sys, datetime as dt
from pathlib import Path
import pandas as pd
import streamlit as st

# Make sure we can import from src/
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
for p in [ROOT, SRC]:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from src.universe import sp500
from src.scan import scan as scan_fn

st.set_page_config(page_title="AI-Trade Scanner", layout="wide")
st.title("🔎 AI-Trade — S&P 500 Scanner")

with st.sidebar:
    st.header("Parameters")
    start_date = st.date_input("Start date", dt.date(2018, 1, 1))
    setups = st.multiselect("Setups", ["breakout","pullback","vcp"],
                            default=["breakout","pullback","vcp"])
    chunk  = st.slider("Batch size (symbols per request)", 10, 80, 25, step=5)
    capital = st.number_input("Account size ($)", value=25000, min_value=1000, step=500)
    risk_pct_ui = st.slider("Risk per trade (%)", 0.10, 2.00, 0.50, step=0.10)
    buffer_bp   = st.slider("Entry buffer (bps)", 0, 50, 10)  # 10 bps = 0.10%
    outdir = st.text_input("Output folder", "app_reports")
    colA, colB = st.columns(2)
    run_btn  = colA.button("▶️ Run scan", use_container_width=True)
    load_btn = colB.button("📄 Load last results", use_container_width=True)

def load_results(folder: str) -> pd.DataFrame | None:
    p = Path(folder) / "watchlist.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            st.error(f"Could not read {p}: {e}")
    return None

if run_btn:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    with st.spinner("Scanning S&P 500…"):
        syms = sp500()  # cached weekly by universe.py
        # Convert UI values
        risk_pct = float(risk_pct_ui) / 100.0
        buffer   = float(buffer_bp)   / 10000.0
        # Run the scan (writes CSV + charts)
        scan_fn(
            syms,
            str(start_date),
            outdir,
            setups,
            chunk_size=int(chunk),
            capital=float(capital),
            risk_pct=risk_pct,
            buffer=buffer,
        )
    st.success("Scan complete ✅")

if run_btn or load_btn:
    df = load_results(outdir)
    if df is None:
        st.info("No results yet. Click **Run scan**.")
    else:
        st.subheader("Results")
        # Basic safety if older CSV lacks new columns
        needed = {"symbol","setup","score","price","date"}
        if not needed.issubset(df.columns):
            st.warning("Results file seems old — missing expected columns. Re-run the scan.")
        # Filters
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            setups_avail = sorted(df["setup"].dropna().unique().tolist())
            setup_filter = st.multiselect("Filter setups", setups_avail, default=setups_avail)
        with c2:
            min_score = float(df["score"].min()) if "score" in df else 0.0
            max_score = float(df["score"].max()) if "score" in df else 100.0
            score_cut = st.slider("Min score", min_score, max_score, min_score)
        with c3:
            sort_by = st.selectbox("Sort by", ["score","symbol","date"])
        # Apply filters
        m = df.copy()
        if "setup" in m: m = m[m["setup"].isin(setup_filter)]
        if "score" in m: m = m[m["score"] >= score_cut]
        m = m.sort_values(sort_by, ascending=(sort_by!="score"))
        st.dataframe(m, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download CSV", data=m.to_csv(index=False).encode("utf-8"),
                           file_name="watchlist.csv", mime="text/csv")

        st.markdown("---")
        st.subheader("Chart preview")
        sym = st.selectbox("Pick a symbol", m["symbol"].unique() if not m.empty else [])
        if sym:
            img = Path(outdir) / "charts" / f"{sym}.png"
            if img.exists():
                st.image(str(img), caption=f"{sym} — {', '.join(m[m.symbol==sym]['setup'].unique())}")
                # Show the plan if present
                cols_to_show = ["entry","stop","risk_per_share","shares_for_plan","position_value","atr14","pivot","price","date","setup","score"]
                plan_cols = [c for c in cols_to_show if c in m.columns]
                st.write(m[m.symbol==sym][plan_cols].head(1).T)
            else:
                st.info("Chart image not found yet for this symbol.")

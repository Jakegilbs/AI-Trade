import os, sys, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .features import compute_features, candidate_mask
except Exception:
    from src.prebreak.features import compute_features, candidate_mask

try:
    from ..universe import us_all, sp500, from_file
    from ..cache import load as cache_load, save as cache_save
    from ..setups import trade_plan_bases
    from ..plotting import save_chart
except Exception:
    from src.universe import us_all, sp500, from_file
    from src.cache import load as cache_load, save as cache_save
    from src.setups import trade_plan_bases
    from src.plotting import save_chart

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    # pick the column level that contains OHLCV
    if hasattr(df, "columns") and hasattr(df.columns, "levels"):
        ohlcv = {"Open","High","Low","Close","Adj Close","Volume"}
        chosen = None
        for lvl in range(df.columns.nlevels):
            vals = set(map(str, df.columns.get_level_values(lvl)))
            if len(ohlcv & vals) >= 4:
                chosen = lvl
                break
        if chosen is not None:
            df = df.copy()
            df.columns = df.columns.get_level_values(chosen)
            df = df.loc[:, ~df.columns.duplicated()]
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    if not keep:
        return pd.DataFrame()
    out = df.loc[:, keep].copy().apply(lambda s: pd.to_numeric(s, errors="coerce"))
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def _robust_download(sym: str, start: str):
    tries = [
        ("download-std", lambda: yf.download(sym, start=start, auto_adjust=False, progress=False, threads=False)),
        ("ticker-history", lambda: yf.Ticker(sym).history(start=start, interval="1d", auto_adjust=False)),
        ("download-max", lambda: yf.download(sym, period="max", auto_adjust=False, progress=False, threads=False)),
    ]
    for _, fn in tries:
        try:
            df = _sanitize(fn())
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(0.2)
    return pd.DataFrame()

def prefetch_to_cache(symbols, start, chunk=25, verbose=True):
    # batch first (fast), then fill misses one-by-one
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i+chunk]
        try:
            if verbose:
                print(f"[prefetch] batch {i//chunk+1}: {len(batch)} symbols")
            downloaded = yf.download(batch, start=start, auto_adjust=False, progress=False, group_by="ticker", threads=False)
        except Exception as e:
            if verbose:
                print(f"[prefetch] chunk error: {e} — fallback to single")
            downloaded = None
        for sym in batch:
            df = cache_load(sym, start)
            if df is None or df.empty:
                df = _slice_multi(downloaded, sym) if downloaded is not None else None
            if df is None or df.empty:
                df = _robust_download(sym, start)
            df = _sanitize(df)
            if not df.empty:
                cache_save(sym, df)

def _slice_multi(downloaded: pd.DataFrame, symbol: str):
    if downloaded is None or not isinstance(downloaded.columns, pd.MultiIndex):
        return None
    for lvl in (1, 0):
        try:
            df = downloaded.xs(symbol, axis=1, level=lvl)
            keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
            if len(keep) >= 4:
                return df[keep]
        except Exception:
            pass
    return None

def load_ohlcv(sym: str, start: str):
    df = cache_load(sym, start)
    if df is None or df.empty:
        df = _robust_download(sym, start)
    df = _sanitize(df)
    if df.empty:
        return None
    cache_save(sym, df)
    return df

def score_universe(model_path="models/prebreak_model.joblib", start="2018-01-01", universe="sp500", symbols_file="data/universe_custom.txt", out_dir="reports_pre", proba_threshold=0.65, min_price=5.0, min_addv=3_000_000.0, capital=25_000.0, risk_pct=0.005, buffer=0.001):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    charts_dir = os.path.join(out_dir, "charts")
    Path(charts_dir).mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)
    with open("models/prebreak_meta.json","r") as f:
        meta = json.load(f)
    feat_cols = meta["features"]

    # choose universe
    if universe == "all":
        symbols = us_all()
    elif universe == "file":
        symbols = from_file(symbols_file)
    else:
        symbols = sp500()

    # prefetch everything up-front to avoid slow per-ticker downloads
    print(f"[prefetch] universe={universe} symbols={len(symbols)}")
    prefetch_to_cache(["SPY"], start, chunk=1, verbose=False)
    prefetch_to_cache(symbols, start, chunk=25, verbose=True)

    spy = load_ohlcv("SPY", start)
    if spy is None or spy.empty:
        raise RuntimeError("SPY data missing")
    spy_close = spy["Close"]

    rows = []
    for idx, sym in enumerate(symbols, 1):
        if idx % 50 == 0:
            print(f"[score] {idx}/{len(symbols)} ...")
        df = load_ohlcv(sym, start)
        if df is None or df.empty or len(df) < 260:
            continue

        feats = compute_features(df, spy_close)
        if feats.empty:
            continue

        last = feats.iloc[-1].copy()
        price = float(last["close"])
        addv20 = float(last.get("addv20", 0) or 0)
        if price < min_price or addv20 < min_addv:
            continue

        # candidate gate (near pivot / contraction, etc.)
        cand = bool(candidate_mask(feats).iloc[-1])
        if not cand:
            continue

        # Build a DataFrame with the SAME feature names used in training
        x_df = last.reindex(feat_cols).astype(float).to_frame().T.fillna(0.0)
        proba = float(model.predict_proba(x_df)[0,1])

        pivot = float(last["pivot"])
        near_pivot = price >= pivot * 0.94

        if proba >= proba_threshold and near_pivot:
            bases = trade_plan_bases(df, "breakout")
            if bases:
                entry = bases["entry_base"] * (1 + buffer)
                stop = min(bases["stop_base"], entry - 0.01)
                risk_per_share = max(entry - stop, 0.01)
                shares = int(math.floor(capital * risk_pct / risk_per_share))
                pos_value = shares * entry
                rows.append(
                    {
                        "symbol": sym,
                        "date": str(df.index[-1].date()),
                        "price": price,
                        "pivot": float(bases["pivot"]),
                        "entry": round(entry,2),
                        "stop": round(stop,2),
                        "shares": shares,
                        "position_value": round(pos_value,2),
                        "proba": round(proba,4),
                    }
                )
                # chart with plan
                try:
                    save_chart(df, sym, [{"setup":"prebreak"}], charts_dir, plan={"entry": entry, "stop": stop})
                except Exception:
                    pass

    out_csv = os.path.join(out_dir, "early_watchlist.csv")
    pd.DataFrame(rows).sort_values(["proba","symbol"], ascending=[False, True]).to_csv(out_csv, index=False)
    return out_csv, charts_dir

if __name__ == "__main__":
    path, charts = score_universe()
    print("saved", path)
    print("charts", charts)

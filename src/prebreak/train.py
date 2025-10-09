import os, sys, json, time
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import joblib
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .features import compute_features, candidate_mask
except Exception:
    from src.prebreak.features import compute_features, candidate_mask

try:
    from ..universe import sp1500, sp500, us_all, from_file
    from ..cache import load as cache_load, save as cache_save
except Exception:
    from src.universe import sp1500, sp500, us_all, from_file
    from src.cache import load as cache_load, save as cache_save

from .label import label_prebreakout

def _chunks(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

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

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
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
    tries = []
    tries.append(("download-std", lambda: yf.download(sym, start=start, auto_adjust=False, progress=False, threads=False)))
    tries.append(("ticker-history", lambda: yf.Ticker(sym).history(start=start, interval="1d", auto_adjust=False)))
    tries.append(("download-max", lambda: yf.download(sym, period="max", auto_adjust=False, progress=False, threads=False)))
    for _, fn in tries:
        try:
            df = _sanitize(fn())
            if not df.empty:
                return df
        except Exception:
            pass
        time.sleep(0.25)
    return pd.DataFrame()

def prefetch_to_cache(symbols: List[str], start: str, chunk: int = 20, verbose: bool = True):
    fetched = 0
    for batch in _chunks(symbols, chunk):
        try:
            if verbose:
                print(f"[prefetch] downloading {len(batch)} symbols ...")
            downloaded = yf.download(batch, start=start, auto_adjust=False, progress=False, group_by="ticker", threads=False)
        except Exception as e:
            if verbose:
                print(f"[prefetch] chunk error: {e} — falling back to single loads")
            downloaded = None
        for sym in batch:
            try:
                df = cache_load(sym, start)
                if df is None or df.empty:
                    df = _slice_multi(downloaded, sym) if downloaded is not None else None
                if df is None or df.empty:
                    df = _robust_download(sym, start)
                df = _sanitize(df)
                if not df.empty:
                    cache_save(sym, df)
                    fetched += 1
                    if verbose and fetched % 25 == 0:
                        print(f"[prefetch] cached {fetched} symbols so far")
            except Exception as e:
                if verbose:
                    print(f"[prefetch] {sym}: {e}")
    if verbose:
        print(f"[prefetch] done. cached/updated ≈ {fetched}")

def load_ohlcv(sym: str, start: str):
    df = cache_load(sym, start)
    if df is None or df.empty:
        df = _robust_download(sym, start)
    df = _sanitize(df)
    if df.empty:
        return None
    cache_save(sym, df)
    return df

def pick_universe(universe: str, symbols_file: str):
    if universe == "sp500":
        return sp500()
    if universe == "all":
        return us_all()
    if universe == "file":
        return from_file(symbols_file)
    if universe == "sp1500":
        return sp1500()
    return sp500()

def build_dataset(symbols, start="2014-01-01", spy="SPY", verbose=True):
    spy_df = load_ohlcv(spy, start)
    if spy_df is None or spy_df.empty:
        raise RuntimeError("SPY data missing")
    spy_close = spy_df["Close"]

    frames = []
    kept = 0
    skipped_hist = 0
    skipped_feat = 0
    for i, sym in enumerate(symbols, 1):
        try:
            df = load_ohlcv(sym, start)
            if df is None or df.empty or len(df) < 260:
                skipped_hist += 1
                if verbose and i % 25 == 0:
                    print(f"[build] {i}/{len(symbols)} processed | kept {kept} | skipped_hist {skipped_hist} | skipped_feat {skipped_feat}")
                continue
            f = compute_features(df, spy_close)
            if f is None or len(f) < 260:
                skipped_feat += 1
                if verbose and i % 25 == 0:
                    print(f"[build] {i}/{len(symbols)} processed | kept {kept} | skipped_hist {skipped_hist} | skipped_feat {skipped_feat}")
                continue
            # make MultiIndex (symbol, date)
            f = f.copy()
            f.index.name = "date"
            f["symbol"] = sym
            f = f.reset_index().set_index(["symbol","date"]).sort_index()
            frames.append(f)
            kept += 1
            if verbose and (kept % 25 == 0 or i % 50 == 0):
                print(f"[build] {i}/{len(symbols)} processed | kept {kept} | skipped_hist {skipped_hist} | skipped_feat {skipped_feat}")
        except Exception as e:
            if verbose:
                print(f"[build] {sym}: {e}")

    if not frames:
        raise RuntimeError("no data frames collected")

    big = pd.concat(frames).sort_index()

    # labels and candidate mask aligned to MultiIndex
    y = label_prebreakout(big, lookahead=20, price_confirm=0.01, vol_mult=1.15)
    mask = candidate_mask(big)

    # final X/y (drop any non-feature cols; pivot kept only to filter later)
    X = big.loc[mask]
    y = y.loc[X.index]

    # features for model (exclude pivot from training)
    feat_cols = [c for c in X.columns if c not in ["pivot"]]
    X_model = X[feat_cols].astype(float)

    if verbose:
        print(f"[build] assembled {len(big)} rows across {len(frames)} symbols")
        print(f"[build] candidates: {len(X_model)} rows; positives: {int(y.sum())}")
    meta = {"index": list(map(lambda t: f"{t[0]}|{t[1]}", X_model.index))}
    return X_model, y, big, {"features": feat_cols, **meta}

def train_and_save(start="2014-01-01", out_dir="models", universe="sp1500", symbols_file="data/universe_custom.txt", limit=None, prefetch_chunk=20, verbose=True):
    syms_full = pick_universe(universe, symbols_file)
    syms = syms_full[:int(limit)] if limit is not None else syms_full
    if verbose:
        print(f"[train] universe={universe} | symbols={len(syms)} | start={start}")
    # prefetch SPY first (so spy_close is available)
    prefetch_to_cache(["SPY"], start, chunk=1, verbose=verbose)
    prefetch_to_cache(syms, start, chunk=prefetch_chunk, verbose=verbose)

    X, y, big, meta = build_dataset(syms, start=start, verbose=verbose)
    feat_cols = meta["features"]

    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for fold, (tr, va) in enumerate(tscv.split(X), 1):
        m = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=400, l2_regularization=0.0)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:,1]
        auc = roc_auc_score(y.iloc[va], p)
        aucs.append(auc)
        if verbose:
            print(f"[cv] fold {fold} AUC={auc:.4f}")

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=600, l2_regularization=0.0)
    model.fit(X, y)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "prebreak_model.joblib"))
    with open(os.path.join(out_dir, "prebreak_meta.json"), "w") as f:
        json.dump({"features": feat_cols, "cv_auc_mean": float(np.mean(aucs)), "cv_auc_std": float(np.std(aucs)), "start": start, "universe": universe, "n_symbols": len(syms)}, f, indent=2)
    if verbose:
        print(f"[train] saved model → {os.path.join(out_dir, 'prebreak_model.joblib')}")
        print(f"[train] CV AUC mean={np.mean(aucs):.4f} std={np.std(aucs):.4f} | features={len(feat_cols)}")
    return float(np.mean(aucs)), float(np.std(aucs)), feat_cols

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--universe", choices=["sp500","sp1500","all","file"], default="sp1500")
    ap.add_argument("--symbols_file", default="data/universe_custom.txt")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--prefetch_chunk", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_and_save(start=args.start, out_dir=args.out_dir, universe=args.universe, symbols_file=args.symbols_file, limit=args.limit, prefetch_chunk=args.prefetch_chunk, verbose=args.verbose or True)

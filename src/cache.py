from pathlib import Path
import pandas as pd
import numpy as np

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

KEEP_COLS = ["Open","High","Low","Close","Adj Close","Volume"]

def _path(sym: str) -> Path:
    return CACHE_DIR / f"{sym}.csv"

def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df, "columns") and hasattr(df.columns, "levels"):
        ohlcv = set(KEEP_COLS)
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
    return df

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame()
    df = _flatten_ohlcv(df)
    cols = [c for c in KEEP_COLS if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = df.loc[:, cols].copy()
    out = out.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out

def load(sym: str, start_iso: str):
    p = _path(sym)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, index_col=0)
        df = _sanitize(df)
        if df.empty:
            return None
        if df.index.min().strftime("%Y-%m-%d") > start_iso:
            return None
        return df
    except Exception:
        return None

def save(sym: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    out = _sanitize(df)
    if out.empty:
        return
    out.to_csv(_path(sym), date_format="%Y-%m-%d")

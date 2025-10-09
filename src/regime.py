import pandas as pd
import yfinance as yf

def _vix_series(start: str, end: str | None) -> pd.Series:
    """Return a clean Series named 'vix_level' regardless of yfinance shape."""
    df = yf.download("^VIX", start=start, end=end, auto_adjust=False, progress=False)
    if df is None or len(df) == 0:
        return pd.Series(dtype=float, name="vix_level")
    # Handle possible shapes: columns may be simple or MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        # Prefer 'Adj Close' if present
        if ("Adj Close", "^VIX") in df.columns:
            s = df[("Adj Close", "^VIX")]
        elif ("Close", "^VIX") in df.columns:
            s = df[("Close", "^VIX")]
        else:
            s = df.xs("^VIX", axis=1, level=-1).filter(["Adj Close","Close"]).iloc[:,0]
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            # Fallback: take the first numeric column
            s = df.select_dtypes("number").iloc[:,0]
    return s.dropna().rename("vix_level")

def download_vix(start: str, end: str | None) -> pd.Series:
    return _vix_series(start, end)

def trend_regime(px: pd.Series, fast=50, slow=200) -> pd.Series:
    sfast = px.rolling(fast, min_periods=fast).mean()
    sslow = px.rolling(slow, min_periods=slow).mean()
    reg = (sfast > sslow).astype(int)
    return reg.rename("trend_up")

def with_regime_features(X: pd.DataFrame, px: pd.Series, start: str, end: str | None) -> pd.DataFrame:
    vix = download_vix(start, end)  # guaranteed Series named 'vix_level'
    df = X.copy()
    # Join VIX level (ffill to align weekends/holidays)
    df = df.join(vix, how="left").ffill()
    # VIX returns
    df["vix_ret_1"] = vix.pct_change(1).reindex(df.index).fillna(0.0)
    df["vix_ret_5"] = vix.pct_change(5).reindex(df.index).fillna(0.0)
    # Trend regime
    df = df.join(trend_regime(px), how="left").fillna(0)
    return df

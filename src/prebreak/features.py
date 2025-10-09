import numpy as np
import pandas as pd

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def ema(s: pd.Series, span: int) -> pd.Series:
    s = _to_num(s)
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, w: int) -> pd.Series:
    s = _to_num(s)
    return s.rolling(w, min_periods=w).mean()

def atr(df: pd.DataFrame, w: int = 14) -> pd.Series:
    h = _to_num(df["High"]); l = _to_num(df["Low"]); c = _to_num(df["Close"])
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w).mean()

def rolling_slope_fast(y: pd.Series, lookback: int) -> pd.Series:
    y = _to_num(y).interpolate(limit_direction="both")
    idx = y.index
    a = y.to_numpy(dtype=float, copy=True)
    n = len(a); L = int(lookback)
    if n < L or L < 2:
        return pd.Series(index=idx, dtype=float)

    # replace remaining non-finite with local mean to keep math stable
    if not np.isfinite(a).all():
        # cheap fallback: fill with rolling mean over L
        pad = np.nanmean(a[np.isfinite(a)]) if np.isfinite(a).any() else 0.0
        a = np.where(np.isfinite(a), a, pad)

    x = np.arange(L, dtype=float)
    sum_x = L * (L - 1) / 2.0
    sum_x2 = (L * (L - 1) * (2 * L - 1)) / 6.0
    denom = L * sum_x2 - (sum_x ** 2)

    csum = np.cumsum(a)
    sum_y = csum[L-1:] - np.concatenate(([0.0], csum[:-L]))
    sum_xy = np.convolve(a, x[::-1], mode="valid")
    slope_valid = (L * sum_xy - sum_x * sum_y) / denom

    out = np.full(n, np.nan, dtype=float)
    out[L-1:] = slope_valid
    return pd.Series(out, index=idx)

def compute_features(df: pd.DataFrame, spy_close: pd.Series) -> pd.DataFrame:
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close["Close"] if "Close" in spy_close.columns else spy_close.iloc[:, 0]
    spy_close = _to_num(spy_close)

    df = df.copy()
    c = _to_num(df["Close"])
    h = _to_num(df["High"])
    l = _to_num(df["Low"])
    v = _to_num(df["Volume"])

    ret = c.pct_change()
    vol20 = v.rolling(20, min_periods=10).mean()
    addv20 = (c * v).rolling(20, min_periods=10).mean()

    ema10 = ema(c, 10)
    ema20 = ema(c, 20)
    sma50 = sma(c, 50)
    sma200 = sma(c, 200)
    atr14 = atr(df, 14)

    std20 = ret.rolling(20, min_periods=20).std()
    std50 = ret.rolling(50, min_periods=50).std()
    vol_ratio = std20 / std50

    rng60 = (h.rolling(60, min_periods=30).max() - l.rolling(60, min_periods=30).min()) / c
    rng30 = (h.rolling(30, min_periods=20).max() - l.rolling(30, min_periods=20).min()) / c
    rng15 = (h.rolling(15, min_periods=10).max() - l.rolling(15, min_periods=10).min()) / c
    vcp_score = (rng60 - rng30) + (rng30 - rng15)

    high252_excl = c.shift(1).rolling(252, min_periods=120).max()
    pct_to_high = (c / high252_excl) - 1.0

    slope50 = rolling_slope_fast(sma50, 20)

    spy_aligned = spy_close.reindex(df.index).ffill().bfill()
    rs = (c / spy_aligned)
    rs_slope20 = rolling_slope_fast(rs, 20)

    base_days_near_high = (c >= high252_excl * 0.9).rolling(40, min_periods=20).sum()
    tight_close = std20

    f = pd.concat(
        {
            "close": c,
            "volume": v,
            "addv20": addv20,
            "rvol": v / vol20,
            "ema10": ema10,
            "ema20": ema20,
            "sma50": sma50,
            "sma200": sma200,
            "atr14": atr14,
            "std20": std20,
            "std50": std50,
            "vol_ratio": vol_ratio,
            "rng60": rng60,
            "rng30": rng30,
            "rng15": rng15,
            "vcp_score": vcp_score,
            "pct_to_high": pct_to_high,
            "slope50": slope50,
            "rs": rs,
            "rs_slope20": rs_slope20,
            "base_days_near_high": base_days_near_high,
            "tight_close": tight_close,
            "pivot": high252_excl,
        },
        axis=1,
    )

    f = f.replace([np.inf, -np.inf], np.nan).dropna()
    return f

def candidate_mask(features: pd.DataFrame) -> pd.Series:
    cond1 = features["pct_to_high"] > -0.08
    cond2 = features["vol_ratio"] < 1.1
    cond3 = features["base_days_near_high"] >= 10
    return cond1 & cond2 & cond3

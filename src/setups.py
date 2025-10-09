import numpy as np
import pandas as pd

def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def atr(df: pd.DataFrame, w: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(w, min_periods=w).mean()

def _lin_slope(y: pd.Series) -> float:
    y = y.dropna()
    n = len(y)
    if n < 5:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)

def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b

def _local_minima(s: pd.Series, w: int = 3) -> list[int]:
    idx = []
    v = s.values
    n = len(v)
    for i in range(w, n - w):
        left = v[i - w:i]
        right = v[i + 1:i + 1 + w]
        if v[i] < np.min(left) and v[i] < np.min(right):
            idx.append(i)
    return idx

def setup_breakout(df: pd.DataFrame):
    if len(df) < 200:
        return False, 0.0, {}
    h = df["High"]
    c = df["Close"]
    v = df["Volume"]
    prior_max = c.shift(1).rolling(252, min_periods=120).max()
    vol_avg = v.rolling(20, min_periods=10).mean()
    hi_ok = (h.iloc[-1] >= float(prior_max.iloc[-1]) * 0.999) or (c.iloc[-1] >= float(prior_max.iloc[-1]) * 0.999)
    vol_ok = v.iloc[-1] >= 1.15 * float(vol_avg.iloc[-1] or 1)
    close_ok = c.iloc[-1] >= float(prior_max.iloc[-1]) * 0.995
    hit = bool(hi_ok and vol_ok and close_ok)
    base = float(prior_max.iloc[-1] or c.iloc[-2])
    score = float((_pct(float(c.iloc[-1]), base) * 100) + min(40.0, (v.iloc[-1] / (float(vol_avg.iloc[-1] or 1))) * 10))
    ctx = {"prior_52w_high": float(prior_max.iloc[-1] or 0), "vol_x_avg20": float(v.iloc[-1] / (float(vol_avg.iloc[-1] or 1)))}
    return hit, score, ctx

def setup_rounding_bottom(df: pd.DataFrame):
    if len(df) < 160:
        return False, 0.0, {}
    c = df["Close"]
    h = df["High"]
    v = df["Volume"]
    win = 180 if len(df) >= 200 else max(120, len(df) - 20)
    s = ema(c, 10).iloc[-win:]
    vwin = v.iloc[-win:]
    if len(s) < 120:
        return False, 0.0, {}
    t_idx = int(np.argmin(s.values))
    if t_idx < 20 or t_idx > len(s) - 20:
        return False, 0.0, {}
    left = s.iloc[:t_idx]
    right = s.iloc[t_idx + 1:]
    if _lin_slope(left.tail(60)) >= 0:
        return False, 0.0, {}
    if _lin_slope(right.head(60)) <= 0:
        return False, 0.0, {}
    depth = _pct(float(s.iloc[t_idx]), float(max(s.iloc[0], left.max())))
    if depth > -0.08:
        return False, 0.0, {}
    v_decline = vwin.iloc[:t_idx].mean()
    v_trough = vwin.iloc[max(0, t_idx - 10): t_idx + 10].mean()
    v_recent = vwin.iloc[-15:].mean()
    vol_ok = (v_trough <= 0.95 * v_decline) and (v_recent >= 1.05 * v_trough)
    left_lip = float(h.iloc[-win:][:t_idx].max())
    vol20 = v.rolling(20, min_periods=10).mean().iloc[-1]
    near_break = (c.iloc[-1] >= left_lip * 0.995) and (v.iloc[-1] >= 1.10 * float(vol20 or 1))
    hit = bool(vol_ok and near_break)
    score = float((_pct(float(c.iloc[-1]), left_lip) * 100) + min(30.0, (v.iloc[-1] / (float(vol20 or 1))) * 8))
    ctx = {"left_lip": left_lip}
    return hit, score, ctx

def setup_pullback_20ema(df: pd.DataFrame):
    if len(df) < 220:
        return False, 0.0, {}
    c = df["Close"]
    h = df["High"]
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    sma200 = sma(c, 200)
    uptrend = (c > ema50) & (ema50 > sma200)
    near20 = ((c - ema20).abs() / ema20) < 0.01
    reclaim = c > h.shift(1)
    cond = uptrend & near20 & reclaim
    hit = bool(cond.iloc[-1])
    score = float((1 - ((c.iloc[-1] - ema20.iloc[-1]) / ema20.iloc[-1]) ** 2) * 100)
    ctx = {"ema20": float(ema20.iloc[-1] or 0), "ema50": float(ema50.iloc[-1] or 0), "sma200": float(sma200.iloc[-1] or 0)}
    return hit, score, ctx

def setup_vcp(df: pd.DataFrame):
    if len(df) < 120:
        return False, 0.0, {}
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atrp = atr14 / c
    atrp_rank = (atrp / atrp.rolling(120, min_periods=60).max()).fillna(1.0)
    near_high = c / c.rolling(100, min_periods=60).max()
    cond = (atrp_rank < 0.35) & (near_high > 0.95)
    hit = bool(cond.iloc[-1])
    score = float((1 - atrp_rank.iloc[-1]) * 100 + (near_high.iloc[-1] - 0.95) * 200)
    ctx = {"atr_pct": float((atr14.iloc[-1] / c.iloc[-1]) if pd.notna(atr14.iloc[-1]) else 0), "pct_of_100d_high": float(near_high.iloc[-1] or 0)}
    return hit, score, ctx

def setup_inv_hs(df: pd.DataFrame):
    if len(df) < 160:
        return False, 0.0, {}
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]
    win = 160
    s = ema(c, 5).iloc[-win:]
    mins = _local_minima(s, w=3)
    if len(mins) < 3:
        return False, 0.0, {}
    i1, i2, i3 = mins[-3], mins[-2], mins[-1]
    if not (s.iloc[i2] < s.iloc[i1] and s.iloc[i2] < s.iloc[i3] and s.iloc[i3] >= s.iloc[i1] * 0.95 and s.iloc[i3] <= s.iloc[i1] * 1.05):
        return False, 0.0, {}
    mid1 = int((i1 + i2) / 2)
    mid2 = int((i2 + i3) / 2)
    neckline = float(max(h.iloc[-win + i1:-win + mid1].max(), h.iloc[-win + mid2:-1].max()))
    vol20 = v.rolling(20, min_periods=10).mean().iloc[-1]
    hit = (c.iloc[-1] >= neckline * 1.001) and (v.iloc[-1] >= 1.2 * float(vol20 or 1))
    score = float(_pct(float(c.iloc[-1]), neckline) * 100)
    ctx = {"neckline": neckline}
    return bool(hit), score, ctx

SETUPS = {
    "breakout": setup_breakout,
    "rounding": setup_rounding_bottom,
    "pullback": setup_pullback_20ema,
    "vcp": setup_vcp,
    "inv_hs": setup_inv_hs
}

def evaluate(df: pd.DataFrame, setups: list) -> list:
    out = []
    for name in setups:
        fn = SETUPS.get(name)
        if not fn:
            continue
        hit, score, ctx = fn(df)
        if hit:
            row = {"setup": name, "score": round(float(score), 3)}
            for k, v in ctx.items():
                if isinstance(v, (int, float, np.floating)):
                    row[k] = float(v)
                else:
                    row[k] = v
            out.append(row)
    return out

def trade_plan_bases(df: pd.DataFrame, setup: str) -> dict:
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    a = float(atr(df, 14).iloc[-1] or 0)
    if setup == "breakout":
        prior_max = c.shift(1).rolling(252, min_periods=120).max().iloc[-1]
        pivot = float(prior_max if pd.notna(prior_max) else h.iloc[-2])
        entry_base = float(max(h.iloc[-1], pivot))
        stop_struct = float(l.iloc[-1])
        stop_atr = entry_base - 1.1 * a
        stop_base = float(min(stop_struct, stop_atr))
    elif setup == "rounding":
        wb = min(len(df), 180)
        left_lip = float(h.iloc[-wb: -int(wb / 3)].max() if wb >= 60 else h.iloc[-wb:].max())
        pivot = float(left_lip)
        entry_base = float(max(h.iloc[-1], pivot))
        recent_low = float(l.rolling(15, min_periods=5).min().iloc[-1])
        stop_atr = entry_base - 1.25 * a
        stop_base = float(min(recent_low, stop_atr))
    else:
        return {}
    return {"pivot": float(pivot), "atr14": float(a), "entry_base": float(entry_base), "stop_base": float(stop_base)}

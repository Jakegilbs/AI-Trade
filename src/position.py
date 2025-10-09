import numpy as np
import pandas as pd

def proba_to_size(p: pd.Series, slope: float = 4.0) -> pd.Series:
    """Map probability -> [0,1] size. 0.5→0, 0.75→~1 with slope=4."""
    x = (p - 0.5) * slope
    return x.clip(lower=0.0, upper=1.0)

def realized_vol(returns: pd.Series, lookback: int = 20) -> pd.Series:
    """Daily realized vol (annualized)."""
    return returns.rolling(lookback).std() * (252 ** 0.5)

def vol_target_scale(returns: pd.Series, target_ann_vol: float = 0.10, lookback: int = 20) -> pd.Series:
    """Scaling factor so (position * scale) ≈ target annualized vol."""
    rv = realized_vol(returns, lookback=lookback)
    scale = target_ann_vol / rv.replace(0, np.nan)
    return scale.clip(upper=5.0).fillna(0.0)

def apply_sizing(prices: pd.Series, size: pd.Series, base_signal: pd.Series) -> pd.Series:
    """
    Convert base long/flat signal (0/1) and continuous size [0,1] into position.
    Uses previous-day values (no look-ahead).
    """
    pos = base_signal.shift(1).fillna(0) * size.shift(1).fillna(0)
    return pos.clip(0.0, 1.0)

def ema_smooth(series: pd.Series, span: int = 5) -> pd.Series:
    """EMA smoother for sizes/positions to reduce whipsaw turnover."""
    return series.ewm(span=span, adjust=False).mean()

def enforce_min_hold(signal: pd.Series, min_hold_days: int = 3) -> pd.Series:
    """
    Ensure we hold positions at least N days once entered.
    Works on binary (0/1) signals.
    """
    sig = signal.copy().fillna(0).astype(int)
    out = sig.copy()
    holding = 0
    for i in range(len(sig)):
        if holding > 0:
            out.iloc[i] = 1
            holding -= 1
            if sig.iloc[i] == 1:
                holding = max(holding, min_hold_days - 1)
        else:
            if sig.iloc[i] == 1:
                out.iloc[i] = 1
                holding = min_hold_days - 1
            else:
                out.iloc[i] = 0
    out.name = signal.name
    return out

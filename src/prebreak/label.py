import numpy as np
import pandas as pd

def label_prebreakout(df: pd.DataFrame, lookahead: int = 20, price_confirm: float = 0.01, vol_mult: float = 1.15) -> pd.Series:
    c, h, v = df["close"], df["close"], df.get("volume", None)
    pivot = df["pivot"]
    vol20 = df.get("volume", None).rolling(20, min_periods=10).mean() if "volume" in df.columns else None
    y = pd.Series(0, index=df.index, dtype=int)
    for i in range(len(df) - lookahead - 1):
        win = slice(i + 1, i + 1 + lookahead)
        price_ok = (c.iloc[win] >= pivot.iloc[i] * (1.0 + price_confirm))
        if vol20 is not None:
            vol_ok = (df["volume"].iloc[win] >= vol_mult * (vol20.iloc[win].fillna(1.0)))
            hit = bool((price_ok & vol_ok).any())
        else:
            hit = bool(price_ok.any())
        y.iloc[i] = 1 if hit else 0
    return y

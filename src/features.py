import numpy as np
import pandas as pd

def rsi(px: pd.Series, window=14) -> pd.Series:
    d = px.diff()
    up = d.clip(lower=0).rolling(window).mean()
    dn = (-d.clip(upper=0)).rolling(window).mean()
    rs = up / (dn.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def make_features(px: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=px.index)
    X["ret_1"] = px.pct_change(1)
    for w in (3,5,10,20,50,100):
        X[f"ret_{w}"] = px.pct_change(w)
        X[f"vol_{w}"] = px.pct_change().rolling(w).std()
        X[f"rsi_{w}"] = rsi(px, w)
    for w in (5,10,20,50,100,200):
        sma = px.rolling(w).mean()
        X[f"gap_sma_{w}"] = (px - sma) / px
    return X.dropna()

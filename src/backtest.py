import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

def split_time(X, y, split_date):
    X_tr, X_te = X.loc[:split_date], X.loc[split_date:]
    y_tr, y_te = y.loc[:split_date], y.loc[split_date:]
    idx = X_tr.index.intersection(y_tr.index)
    X_tr, y_tr = X_tr.loc[idx].dropna(), y_tr.loc[idx].dropna()
    idx = X_te.index.intersection(y_te.index)
    X_te, y_te = X_te.loc[idx].dropna(), y_te.loc[idx].dropna()
    return X_tr, y_tr, X_te, y_te

def proba_to_signal(proba: pd.Series, threshold=0.5) -> pd.Series:
    return (proba > threshold).astype(int)

def pnl_curve(px: pd.Series, signal: pd.Series, cost_bps=2.0, max_pos=1.0) -> pd.DataFrame:
    rets = px.pct_change().reindex(signal.index)
    pos = signal.shift(1).fillna(0) * max_pos
    gross = pos * rets
    turns = pos.diff().abs().fillna(pos.abs())
    costs = turns * (cost_bps/1e4)
    net = gross - costs
    eq = (1 + net).cumprod()
    return pd.DataFrame({"gross": gross, "net": net, "equity": eq})

def metrics(y_true: pd.Series, proba: pd.Series, sig: pd.Series, pnl: pd.DataFrame) -> dict:
    out = {}
    pred = (proba > 0.5).astype(int)
    out["accuracy"] = accuracy_score(y_true, pred)
    try: out["auc"] = roc_auc_score(y_true, proba)
    except: out["auc"] = float("nan")
    total_ret = pnl["equity"].iloc[-1] - 1
    ann_ret = pnl["equity"].iloc[-1]**(252/len(pnl)) - 1
    ann_vol = pnl["net"].std() * (252**0.5)
    out["total_return"] = float(total_ret)
    out["ann_return"] = float(ann_ret)
    out["ann_vol"] = float(ann_vol)
    out["sharpe"] = float(ann_ret / (ann_vol + 1e-9))
    return out

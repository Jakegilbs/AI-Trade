import numpy as np
import pandas as pd

def cagr(equity: pd.Series) -> float:
    days = len(equity)
    if days < 2: return 0.0
    return float(equity.iloc[-1] ** (252/days) - 1)

def max_drawdown(equity: pd.Series) -> float:
    roll = equity.cummax()
    dd = equity/roll - 1
    return float(dd.min())

def sortino(returns: pd.Series, rf=0.0) -> float:
    dr = returns[returns < 0]
    if len(dr) == 0: return 0.0
    ann_ret = returns.mean() * 252
    dd = dr.std() * (252 ** 0.5)
    return float((ann_ret - rf) / (dd + 1e-9))

def turnover(positions: pd.Series) -> float:
    return float(positions.diff().abs().sum())

def summarize(pnl: pd.DataFrame, positions: pd.Series) -> dict:
    eq = pnl["equity"]
    ret = pnl["net"]
    ann_ret = eq.iloc[-1] ** (252/len(eq)) - 1
    ann_vol = ret.std() * (252 ** 0.5)
    sharpe = ann_ret / (ann_vol + 1e-9)
    return {
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": float(cagr(eq)),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino(ret)),
        "max_dd": float(max_drawdown(eq)),
        "turnover": float(turnover(positions))
    }

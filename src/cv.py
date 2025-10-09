from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from .model import build_model, fit_model
from .backtest import pnl_curve
from .position import proba_to_size, apply_sizing
from .config import Config

@dataclass
class CVResult:
    best_params: Dict[str, Any]
    history: pd.DataFrame

def score_fold(y_true: pd.Series, proba: pd.Series, prices: pd.Series, cfg: Config) -> Tuple[float, float]:
    try:
        auc = roc_auc_score(y_true, proba)
    except:
        auc = 0.5
    # base long/flat
    sig = (proba > cfg.proba_threshold).astype(int)
    size = proba_to_size(proba, slope=4.0)
    pos = apply_sizing(prices.reindex(proba.index), size, sig)
    pnl = pnl_curve(prices.reindex(proba.index), pos, cfg.tx_cost_bps, 1.0)
    # OOS Sharpe
    net = pnl["net"]
    ann_ret = (pnl["equity"].iloc[-1])**(252/len(pnl)) - 1 if len(pnl) > 0 else 0
    ann_vol = net.std() * (252 ** 0.5) if len(net) > 2 else 1
    sharpe = ann_ret / (ann_vol + 1e-9)
    return auc, sharpe

def timeseries_cv(X: pd.DataFrame, y: pd.Series, prices: pd.Series, cfg: Config) -> CVResult:
    # Small grid for RandomForest; extend as needed
    grid = {
        "clf__n_estimators": [300, 600],
        "clf__min_samples_leaf": [2, 4, 8],
        "clf__max_depth": [None, 6, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5, test_size=max(60, len(X)//12))
    records = []

    best = None
    best_score = -1

    for n in grid["clf__n_estimators"]:
        for leaf in grid["clf__min_samples_leaf"]:
            for depth in grid["clf__max_depth"]:
                fold_scores = []
                for tr_idx, te_idx in tscv.split(X):
                    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
                    model = build_model()
                    # set params
                    model.set_params(**{
                        "clf__n_estimators": n,
                        "clf__min_samples_leaf": leaf,
                        "clf__max_depth": depth
                    })
                    model = fit_model(model, X_tr, y_tr)
                    proba = pd.Series(model.predict_proba(X_te)[:,1], index=X_te.index)
                    auc, shrp = score_fold(y_te, proba, prices, cfg)
                    fold_scores.append((auc, shrp))
                # combine: weighted 0.7 AUC + 0.3 Sharpe (shift Sharpe to similar scale)
                auc_mean = float(np.mean([a for a,_ in fold_scores]))
                sharpe_mean = float(np.mean([s for _,s in fold_scores]))
                combo = 0.7*auc_mean + 0.3*(0.5 + sharpe_mean/2.0)
                records.append({"n_estimators": n, "min_leaf": leaf, "max_depth": depth,
                                "auc": auc_mean, "sharpe": sharpe_mean, "score": combo})
                if combo > best_score:
                    best_score = combo
                    best = {"clf__n_estimators": n, "clf__min_samples_leaf": leaf, "clf__max_depth": depth}
    hist = pd.DataFrame.from_records(records).sort_values("score", ascending=False)
    return CVResult(best_params=best, history=hist)

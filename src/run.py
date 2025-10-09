import pandas as pd
from pathlib import Path
from config import Config
from data import download_prices
from features import make_features
from labels import make_binary_labels
from model import build_model, fit_model
from backtest import split_time, pnl_curve
from position import proba_to_size, apply_sizing, vol_target_scale
from metrics_ext import summarize
from regime import with_regime_features
from cv import timeseries_cv

def main():
    cfg = Config()
    prices = download_prices(cfg.tickers, cfg.start, cfg.end)
    s = prices.iloc[:, 0]

    # Base features
    X = make_features(s)
    # Regime features (VIX + trend)
    X = with_regime_features(X, s, cfg.start, cfg.end)

    y = make_binary_labels(s, cfg.horizon)
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    # Split
    X_tr, y_tr, X_te, y_te = split_time(X, y, cfg.split_date)
    s_te = s.reindex(X_te.index)

    # ---- CV to pick RF hyperparams (expanding TSCV) ----
    cvres = timeseries_cv(X_tr, y_tr, s.reindex(X_tr.index), cfg)
    best = cvres.best_params
    print("\nBest CV params:", best)
    Path("models").mkdir(exist_ok=True)
    cvres.history.to_csv("models/cv_history.csv", index=False)

    # Train with best params
    model = build_model()
    model.set_params(**best)
    model = fit_model(model, X_tr, y_tr)

    # Predict
    proba = pd.Series(model.predict_proba(X_te)[:, 1], index=X_te.index, name="p_up")

    # Base long/flat & continuous sizing
    base_sig = (proba > cfg.proba_threshold).astype(int)
    size = proba_to_size(proba, slope=4.0)

    # Vol targeting (scale positions so portfolio vol ~ 10% ann.)
    daily_rets = s.pct_change().reindex(X.index)
    scale = vol_target_scale(daily_rets, target_ann_vol=0.10, lookback=20).reindex(proba.index).fillna(0.0)
    pos = apply_sizing(s_te, size, base_sig) * scale

    # Backtest on test set
    pnl = pnl_curve(s_te, pos, cfg.tx_cost_bps, 1.0)
    summ = summarize(pnl, pos)

    print("\n=== MODEL PERFORMANCE (OOS) ===")
    for k, v in summ.items():
        print(f"{k:>12}: {v:.4f}")

    # Feature importance (permutation)
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
        fi = pd.Series(r.importances_mean, index=X_te.columns).sort_values(ascending=False)
    except Exception as e:
        print("Permutation importance failed:", e)
        fi = pd.Series(dtype=float)

    # Save artifacts
    Path("models").mkdir(exist_ok=True)
    pnl.to_csv("models/equity_oos.csv")
    if len(fi) > 0:
        fi.to_csv("models/feature_importance.csv")
        print("Saved feature_importance.csv")
    print("Saved cv_history.csv and equity_oos.csv in models/")

if __name__ == "__main__":
    main()

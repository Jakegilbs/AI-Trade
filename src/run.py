import pandas as pd
from pathlib import Path
from config import Config
from data import download_prices
from features import make_features
from labels import make_binary_labels
from model import build_model, fit_model
from backtest import split_time, proba_to_signal, pnl_curve, metrics

def main():
    cfg = Config()
    prices = download_prices(cfg.tickers, cfg.start, cfg.end)
    s = prices.iloc[:, 0]
    X = make_features(s)
    y = make_binary_labels(s, cfg.horizon)
    X, y = X.align(y, join="inner")
    X_tr, y_tr, X_te, y_te = split_time(X, y, cfg.split_date)

    model = build_model()
    model = fit_model(model, X_tr, y_tr)

    proba = pd.Series(model.predict_proba(X_te)[:,1], index=X_te.index, name="p_up")
    sig = proba_to_signal(proba, cfg.proba_threshold)
    pnl = pnl_curve(s.reindex(sig.index), sig, cfg.tx_cost_bps, cfg.max_position)

    m = metrics(y_te.reindex(proba.index), proba, sig, pnl)
    print("=== Metrics ===")
    for k,v in m.items():
        print(f"{k:>14}: {v:.4f}" if isinstance(v, float) else f"{k:>14}: {v}")
    Path("models").mkdir(exist_ok=True)

if __name__ == "__main__":
    main()

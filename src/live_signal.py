import math, argparse, json
from datetime import datetime
import pandas as pd
from pathlib import Path
from .config import Config
from .data import download_prices
from .features import make_features
from .labels import make_binary_labels
from .model import build_model, fit_model
from .position import proba_to_size, vol_target_scale, apply_sizing, enforce_min_hold, ema_smooth
from .regime import with_regime_features
from .cv import timeseries_cv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--split", default="2021-01-01")
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--target_vol", type=float, default=0.08)   # 8% ann vol
    ap.add_argument("--min_hold", type=int, default=3)
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--slope", type=float, default=3.0)         # proba->size slope
    args = ap.parse_args()

    cfg = Config(tickers=(args.ticker,), start=args.start, split_date=args.split, proba_threshold=args.threshold)

    # 1) Data + features (+ regimes)
    prices = download_prices(cfg.tickers, cfg.start, cfg.end)
    s = prices.iloc[:, 0]
    X = with_regime_features(make_features(s), s, cfg.start, cfg.end)
    y = make_binary_labels(s, cfg.horizon)
    X, y = X.align(y, join="inner", axis=0)

    # 2) CV on train window, fit best params
    X_tr = X.loc[:cfg.split_date]
    y_tr = y.loc[:cfg.split_date]
    cvres = timeseries_cv(X_tr, y_tr, s.loc[:cfg.split_date], cfg)
    model = build_model()
    model.set_params(**cvres.best_params)
    model = fit_model(model, X_tr, y_tr)

    # 3) Probabilities on ALL available dates (so min-hold/smoothing have context)
    proba = pd.Series(model.predict_proba(X)[:,1], index=X.index, name="p_up")

    # 4) Build signals & sizing (with min-hold + smoothing)
    base_sig = (proba > cfg.proba_threshold).astype(int)
    base_sig = enforce_min_hold(base_sig, min_hold_days=args.min_hold)
    size = ema_smooth(proba_to_size(proba, slope=args.slope), span=5)

    # 5) Vol targeting
    dly = s.pct_change().reindex(X.index)
    scale = vol_target_scale(dly, target_ann_vol=args.target_vol).reindex(proba.index).fillna(0.0)

    # 6) Position series and today's target
    pos = apply_sizing(s, size, base_sig) * scale
    date = pos.index[-1]
    weight = float(pos.iloc[-1])
    price = float(s.iloc[-1])
    shares = int(math.floor((args.capital * weight) / price))

    # 7) Emit ticket
    side = "BUY" if weight > 0 else "CLOSE/FLAT"
    out = {
        "date": str(date.date()),
        "ticker": args.ticker,
        "price": round(price, 4),
        "p_up": round(float(proba.iloc[-1]), 4),
        "threshold": args.threshold,
        "target_weight": round(weight, 4),
        "shares_for_capital": shares,
        "capital": args.capital,
        "cv_best_params": cvres.best_params
    }

    Path("models").mkdir(exist_ok=True)
    fn_csv = f"models/orders_{date.date()}.csv"
    fn_json = f"models/live_run_{date.date()}.json"

    pd.DataFrame([{
        "symbol": args.ticker,
        "side": "buy" if shares>0 else "close",
        "shares": shares,
        "target_weight": weight,
        "price_ref": price
    }]).to_csv(fn_csv, index=False)

    with open(fn_json, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== LIVE SIGNAL ===")
    print(f"Date: {out['date']}  Ticker: {out['ticker']}")
    print(f"p_up: {out['p_up']}  threshold: {out['threshold']}")
    print(f"Target weight: {weight:.2%}  Price: ${price:.2f}")
    print(f"Suggested shares for ${args.capital:,.0f}: {shares} ({side})")
    print(f"Saved: {fn_csv} and {fn_json}")

if __name__ == "__main__":
    main()

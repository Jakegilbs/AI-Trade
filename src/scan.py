import argparse, os, sys, time, math, json
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from .setups import evaluate, trade_plan_bases
    from .plotting import save_chart
    from .universe import sp500, us_all, from_file
    from .cache import load as cache_load, save as cache_save
except Exception:
    from src.setups import evaluate, trade_plan_bases
    from src.plotting import save_chart
    from src.universe import sp500, us_all, from_file
    from src.cache import load as cache_load, save as cache_save

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Open","High","Low","Close","Adj Close","Volume"]
    df = df[cols].rename(columns={"Adj Close":"AdjClose"})
    return df.dropna()

def _slice_multi(downloaded: pd.DataFrame, symbol: str):
    if downloaded is None or not isinstance(downloaded.columns, pd.MultiIndex):
        return None
    try:
        df = downloaded.xs(symbol, axis=1, level=1)
    except Exception:
        try:
            df = downloaded.xs(symbol, axis=1, level=0)
        except Exception:
            return None
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[keep] if len(keep) >= 4 else None

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _download_single(sym, start):
    return yf.download(sym, start=start, auto_adjust=False, progress=False, group_by="ticker", threads=False)

def _download_chunk(symbols, start, tries=2, sleep_base=1.0):
    last_exc = None
    for t in range(tries):
        try:
            return yf.download(symbols, start=start, auto_adjust=False, progress=False, group_by="ticker", threads=False)
        except Exception as e:
            last_exc = e
            time.sleep(sleep_base * (2**t))
    raise last_exc

def plan_and_size(df: pd.DataFrame, setup: str, capital: float, risk_pct: float, buffer: float):
    bases = trade_plan_bases(df, setup)
    if not bases:
        return {}
    entry_base = bases["entry_base"]
    entry = entry_base * (1 + buffer)
    stop = min(bases["stop_base"], entry - 0.01)
    atr14 = bases["atr14"]
    risk_per_share = max(entry - stop, 0.01)
    dollar_risk = capital * risk_pct
    shares = int(math.floor(dollar_risk / risk_per_share)) if risk_per_share > 0 else 0
    pos_value = shares * entry
    return {"entry": round(entry, 2), "stop": round(stop, 2), "risk_per_share": round(risk_per_share, 2), "atr14": round(atr14, 2), "pivot": round(bases["pivot"], 2), "shares_for_plan": shares, "position_value": round(pos_value, 2)}

def _near_breakout(df: pd.DataFrame, max_gap=0.015, vol_mult=1.05):
    c, h, v = df["Close"], df["High"], df["Volume"]
    if len(df) < 200:
        return None
    prior_max = c.shift(1).rolling(252, min_periods=120).max().iloc[-1]
    if not np.isfinite(prior_max) or prior_max == 0:
        return None
    gap = float(c.iloc[-1] / prior_max - 1.0)
    vol20 = float(v.rolling(20, min_periods=10).mean().iloc[-1] or 0.0)
    mult = float((v.iloc[-1] / vol20) if vol20 else np.nan)
    if (gap >= -max_gap) and (mult >= vol_mult):
        return {"prior_high": float(prior_max), "gap_to_high_pct": round(gap*100, 2), "vol_mult_20d": round(mult, 2)}
    return None

def _near_rounding(df: pd.DataFrame, window=180, near_pct=0.02, vol_mult=1.02):
    if len(df) < 160:
        return None
    h, c, v = df["High"], df["Close"], df["Volume"]
    w = min(window, len(df))
    left = h.iloc[-w: -w//3] if w >= 60 else h.iloc[-w:]
    if left.empty:
        return None
    left_lip = float(left.max())
    vol20 = float(v.rolling(20, min_periods=10).mean().iloc[-1] or 0.0)
    mult = float((v.iloc[-1] / vol20) if vol20 else np.nan)
    if (c.iloc[-1] >= left_lip * (1 - near_pct)) and (mult >= vol_mult):
        return {"left_lip": left_lip, "gap_to_lip_pct": round((c.iloc[-1]/left_lip - 1)*100, 2), "vol_mult_20d": round(mult, 2)}
    return None

def scan(symbols, start: str, outdir: str, setups: list, chunk_size: int, capital: float, risk_pct: float, buffer: float, min_price: float, min_addv: float, near_gap_breakout: float, near_vol_breakout: float, near_pct_round: float, near_vol_round: float):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    charts_dir = os.path.join(outdir, "charts")
    Path(charts_dir).mkdir(parents=True, exist_ok=True)

    matches = []
    near_rows = []
    audit_rows = []
    total = len(symbols)
    done = 0
    cnt_cached = 0
    cnt_downloaded = 0
    cnt_short = 0
    cnt_liq = 0
    cnt_eval = 0
    cnt_hit = 0
    cnt_err = 0

    for chunk in _chunks(symbols, chunk_size):
        batch = None
        try:
            batch = _download_chunk(chunk, start)
        except Exception:
            batch = None
        for sym in chunk:
            done += 1
            try:
                df = cache_load(sym, start) or _slice_multi(batch, sym)
                if df is None or df.empty:
                    df = _download_single(sym, start)
                    cnt_downloaded += 1
                else:
                    cnt_cached += 1
                if df is None or df.empty:
                    audit_rows.append({"symbol": sym, "status": "nodata"})
                    continue
                df = clean_df(df)
                cache_save(sym, df)
                if len(df) < 120:
                    cnt_short += 1
                    audit_rows.append({"symbol": sym, "status": "short_history"})
                    continue
                price_now = float(df["Close"].iloc[-1])
                addv20 = float((df["Close"] * df["Volume"]).rolling(20, min_periods=10).mean().iloc[-1] or 0)
                if price_now < min_price or addv20 < min_addv:
                    cnt_liq += 1
                    audit_rows.append({"symbol": sym, "status": "filtered_liquidity", "price": price_now, "addv20": round(addv20,2)})
                    if done % 50 == 0:
                        print(f"[{done}/{total}] ...")
                    continue
                cnt_eval += 1
                found = evaluate(df, setups)
                if found:
                    price = price_now
                    date = str(df.index[-1].date())
                    for h in found:
                        plan = plan_and_size(df, h["setup"], capital, risk_pct, buffer)
                        row = {"symbol": sym, "date": date, "price": price, "addv20": round(addv20,2), **h, **plan, "capital": capital, "risk_pct": risk_pct}
                        matches.append(row)
                        save_chart(df, sym, [h], charts_dir, plan=plan)
                    cnt_hit += 1
                    print(f"[{done}/{total}] {sym}: MATCH -> {[h['setup'] for h in found]}")
                    audit_rows.append({"symbol": sym, "status": "hit", "setups": ",".join([h["setup"] for h in found])})
                else:
                    nb = _near_breakout(df, max_gap=near_gap_breakout, vol_mult=near_vol_breakout) if "breakout" in setups else None
                    nr = _near_rounding(df, near_pct=near_pct_round, vol_mult=near_vol_round) if "rounding" in setups else None
                    if nb:
                        nb.update({"symbol": sym, "type": "near_breakout", "price": price_now})
                        near_rows.append(nb)
                    if nr:
                        nr.update({"symbol": sym, "type": "near_rounding", "price": price_now})
                        near_rows.append(nr)
                    if done % 50 == 0:
                        print(f"[{done}/{total}] ...")
                    audit_rows.append({"symbol": sym, "status": "no_hit"})
            except KeyboardInterrupt:
                raise
            except Exception as e:
                cnt_err += 1
                audit_rows.append({"symbol": sym, "status": "error", "error": str(e)})
    res = pd.DataFrame(matches)
    out_csv = os.path.join(outdir, "watchlist.csv")
    if not res.empty:
        sort_cols = [c for c in ["setup","score","symbol"] if c in res.columns]
        asc = [True, False, True][:len(sort_cols)]
        if sort_cols:
            res = res.sort_values(sort_cols, ascending=asc)
        res.to_csv(out_csv, index=False)
        print(f"Saved {len(res)} matches → {out_csv}")
    else:
        res.to_csv(out_csv, index=False)
        print(f"No matches found. Wrote empty CSV to {out_csv}")
    near_csv = os.path.join(outdir, "near_hits.csv")
    if near_rows:
        ndf = pd.DataFrame(near_rows)
        sort_keys = [c for c in ["type","gap_to_high_pct","gap_to_lip_pct"] if c in ndf.columns]
        ndf.sort_values(sort_keys, inplace=True)
        ndf.to_csv(near_csv, index=False)
        print(f"Saved near hits → {near_csv}")
    audit_csv = os.path.join(outdir, "audit.csv")
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)
    summary = {"total_symbols": total, "cached_used": cnt_cached, "downloaded": cnt_downloaded, "short_history": cnt_short, "filtered_liquidity": cnt_liq, "evaluated": cnt_eval, "hits": cnt_hit, "errors": cnt_err, "outputs": {"watchlist": out_csv, "near_hits": near_csv, "audit": audit_csv, "charts_dir": charts_dir}}
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--setups", default="rounding,breakout")
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--capital", type=float, default=25000.0)
    ap.add_argument("--risk_pct", type=float, default=0.005)
    ap.add_argument("--buffer", type=float, default=0.001)
    ap.add_argument("--universe", choices=["sp500","all","file"], default="all")
    ap.add_argument("--symbols_file", default="data/universe_custom.txt")
    ap.add_argument("--min_price", type=float, default=5.0)
    ap.add_argument("--min_addv", type=float, default=3000000.0)
    ap.add_argument("--near_gap_breakout", type=float, default=0.015)
    ap.add_argument("--near_vol_breakout", type=float, default=1.05)
    ap.add_argument("--near_pct_round", type=float, default=0.02)
    ap.add_argument("--near_vol_round", type=float, default=1.02)
    args = ap.parse_args()
    if args.universe == "sp500":
        symbols = sp500()
    elif args.universe == "all":
        symbols = us_all()
    else:
        symbols = from_file(args.symbols_file)
    setups = [s.strip().lower() for s in args.setups.split(",") if s.strip()]
    scan(symbols, args.start, args.out, setups, chunk_size=args.chunk, capital=args.capital, risk_pct=args.risk_pct, buffer=args.buffer, min_price=args.min_price, min_addv=args.min_addv, near_gap_breakout=args.near_gap_breakout, near_vol_breakout=args.near_vol_breakout, near_pct_round=args.near_pct_round, near_vol_round=args.near_vol_round)

if __name__ == "__main__":
    main()

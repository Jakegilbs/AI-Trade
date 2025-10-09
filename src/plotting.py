import os
import pandas as pd

def save_chart(df: pd.DataFrame, symbol: str, matches, outdir: str, lookback: int = 220, plan: dict | None = None):
    os.makedirs(outdir, exist_ok=True)
    data = df.iloc[-lookback:].copy()
    entry = plan.get("entry") if plan else None
    stop = plan.get("stop") if plan else None
    try:
        import mplfinance as mpf
        addplots = []
        if entry is not None:
            entry_line = pd.Series(float(entry), index=data.index)
            addplots.append(mpf.make_addplot(entry_line, width=1.0))
        if stop is not None:
            stop_line = pd.Series(float(stop), index=data.index)
            addplots.append(mpf.make_addplot(stop_line, width=1.0))
        mav = (20, 50, 200)
        title = f"{symbol} | " + ", ".join([m["setup"] for m in matches]) if matches else symbol
        mpf.plot(data, type="candle", volume=True, mav=mav, addplot=addplots if addplots else None, title=title, style="yahoo", savefig=dict(fname=os.path.join(outdir, f"{symbol}.png"), dpi=120, bbox_inches="tight"))
    except Exception:
        import matplotlib.pyplot as plt
        ax = data["Close"].plot(figsize=(10, 4))
        if entry is not None:
            ax.axhline(float(entry), linestyle="--")
        if stop is not None:
            ax.axhline(float(stop), linestyle=":")
        ax.set_title(f"{symbol}")
        fig = ax.get_figure()
        fig.savefig(os.path.join(outdir, f"{symbol}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

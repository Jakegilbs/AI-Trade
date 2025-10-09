from pathlib import Path
import pandas as pd
import yfinance as yf

def download_prices(tickers: tuple[str, ...], start: str, end: str | None = None, cache_dir="data") -> pd.DataFrame:
    Path(cache_dir).mkdir(exist_ok=True)
    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"].rename_axis(columns="ticker")
    else:
        df = df.rename("Close").to_frame().rename(columns={"Close": tickers[0]})
    df = df.dropna(how="all").ffill().dropna()
    out = Path(cache_dir) / f"prices_{'_'.join(tickers)}.csv"
    df.to_csv(out)
    return df

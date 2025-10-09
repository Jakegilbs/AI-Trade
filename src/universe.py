import re, time, io
from pathlib import Path
import pandas as pd
import requests

DATA_DIR = Path("data"); DATA_DIR.mkdir(exist_ok=True)
CACHE_ALL = DATA_DIR / "us_tickers.csv"
UA = {"User-Agent": "Mozilla/5.0"}

def _clean_symbols(syms):
    out = []
    for s in syms:
        if not isinstance(s, str): continue
        s = s.strip().upper()
        if not s: continue
        s = s.replace(".", "-")
        if not re.match(r"^[A-Z0-9\\-]+$", s): continue
        if s.endswith("^"): continue
        out.append(s)
    return sorted(list(dict.fromkeys(out)))

def _get_csv(url, sep=None, tries=3, sleep=1.5):
    for t in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            if sep is None:
                return pd.read_csv(io.StringIO(r.text))
            return pd.read_csv(io.StringIO(r.text), sep=sep)
        except Exception:
            time.sleep(sleep * (2**t))
    return None

def _get_html_tables(url, match=None, tries=3, sleep=1.5):
    for t in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=20)
            r.raise_for_status()
            return pd.read_html(io.StringIO(r.text), match=match)
        except Exception:
            time.sleep(sleep * (2**t))
    return []

def _nasdaq_all():
    a = _get_csv("https://www.nasdaqtrader.com/dynamic/SymbolDirectory/nasdaqtraded.txt", sep="|")
    b = _get_csv("https://www.nasdaqtrader.com/dynamic/SymbolDirectory/otherlisted.txt", sep="|")
    syms = []
    try:
        if a is not None:
            ca = a.copy()
            ca.columns = [c.lower().strip().replace(" ", "_") for c in ca.columns]
            if "etf" in ca.columns: ca = ca[ca["etf"].astype(str).str.upper() != "Y"]
            if "test_issue" in ca.columns: ca = ca[ca["test_issue"].astype(str).str.upper() != "Y"]
            col = "nasdaq_symbol" if "nasdaq_symbol" in ca.columns else ("symbol" if "symbol" in ca.columns else ca.columns[0])
            syms += ca[col].astype(str).tolist()
        if b is not None:
            cb = b.copy()
            cb.columns = [c.lower().strip().replace(" ", "_") for c in cb.columns]
            if "etf" in cb.columns: cb = cb[cb["etf"].astype(str).str.upper() != "Y"]
            if "test_issue" in cb.columns: cb = cb[cb["test_issue"].astype(str).str.upper() != "Y"]
            colb = "act_symbol" if "act_symbol" in cb.columns else ("symbol" if "symbol" in cb.columns else cb.columns[0])
            syms += cb[colb].astype(str).tolist()
    except Exception:
        pass
    return _clean_symbols(syms)

def _wiki_symbols(url, symbol_col="Symbol"):
    tables = _get_html_tables(url, match="Symbol")
    if not tables:
        return []
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        if symbol_col in cols:
            return _clean_symbols(t[symbol_col].astype(str).tolist())
    return []

def sp500():
    p = DATA_DIR / "sp500.csv"
    syms = _wiki_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol")
    if syms:
        pd.DataFrame({"symbol": syms}).to_csv(p, index=False)
        return syms
    if p.exists():
        return pd.read_csv(p)["symbol"].astype(str).tolist()
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","BRK-B","UNH","XOM","JPM","V","TSLA","LLY","MA","AVGO"]

def sp400():
    p = DATA_DIR / "sp400.csv"
    syms = _wiki_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Symbol")
    if syms:
        pd.DataFrame({"symbol": syms}).to_csv(p, index=False)
        return syms
    if p.exists():
        return pd.read_csv(p)["symbol"].astype(str).tolist()
    return []

def sp600():
    p = DATA_DIR / "sp600.csv"
    syms = _wiki_symbols("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Symbol")
    if syms:
        pd.DataFrame({"symbol": syms}).to_csv(p, index=False)
        return syms
    if p.exists():
        return pd.read_csv(p)["symbol"].astype(str).tolist()
    return []

def sp1500():
    return _clean_symbols(sp500() + sp400() + sp600())

def us_all():
    try:
        syms = _nasdaq_all()
        if len(syms) >= 1000:
            pd.DataFrame({"symbol": syms}).to_csv(CACHE_ALL, index=False)
            return syms
    except Exception:
        pass
    sp = sp1500()
    if sp:
        pd.DataFrame({"symbol": sp}).to_csv(CACHE_ALL, index=False)
        return sp
    if CACHE_ALL.exists():
        return pd.read_csv(CACHE_ALL)["symbol"].astype(str).tolist()
    return []

def from_file(path: str):
    p = Path(path)
    if not p.exists(): return []
    try:
        df = pd.read_csv(p)
        if "symbol" in df.columns:
            return _clean_symbols(df["symbol"].astype(str).tolist())
        return _clean_symbols(df.iloc[:,0].astype(str).tolist())
    except Exception:
        with open(p, "r") as f:
            syms = [line.strip() for line in f if line.strip()]
        return _clean_symbols(syms)

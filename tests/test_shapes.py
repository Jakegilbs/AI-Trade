import pandas as pd
from config import Config
from data import download_prices
from features import make_features
from labels import make_binary_labels

def test_shapes():
    cfg = Config()
    df = download_prices(cfg.tickers, cfg.start, cfg.end)
    s = df.iloc[:,0]
    X = make_features(s)
    y = make_binary_labels(s, cfg.horizon)
    X, y = X.align(y, join="inner")
    assert len(X) == len(y) and len(X) > 200

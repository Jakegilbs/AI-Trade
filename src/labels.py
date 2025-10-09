import pandas as pd

def make_binary_labels(px: pd.Series, horizon=1) -> pd.Series:
    fwd = px.pct_change(horizon).shift(-horizon)
    return (fwd > 0).astype(int).rename("y")

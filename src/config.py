from dataclasses import dataclass

@dataclass
class Config:
    tickers: tuple = ("SPY",)
    start: str = "2012-01-01"
    end: str | None = None
    horizon: int = 1
    split_date: str = "2021-01-01"
    tx_cost_bps: float = 2.0
    max_position: float = 1.0
    proba_threshold: float = 0.55

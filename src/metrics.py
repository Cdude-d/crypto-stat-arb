# src/metrics.py
import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, periods_per_year: int) -> float:
    r = returns.dropna()
    if r.std(ddof=0) == 0:
        return 0.0
    return (r.mean() / r.std(ddof=0)) * np.sqrt(periods_per_year)

def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return dd.min()

def hit_rate(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) == 0:
        return 0.0
    return (r > 0).mean()

def summarize(out_df: pd.DataFrame, timeframe: str) -> dict:
    # crude mapping; adjust if you change timeframe
    periods = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
    ppy = periods.get(timeframe, 365)

    ret = out_df["ret_net"]
    eq = out_df["equity"]

    return {
        "Sharpe": float(sharpe(ret, ppy)),
        "MaxDrawdown": float(max_drawdown(eq)),
        "HitRate": float(hit_rate(ret)),
        "TotalReturn": float(eq.iloc[-1] - 1.0),
        "Bars": int(len(out_df)),
    }

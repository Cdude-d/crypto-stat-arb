# src/signals.py
import numpy as np
import pandas as pd

def rolling_ols_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Rolling hedge ratio beta from OLS with intercept:
      y_t = a + beta * x_t + eps_t
    Uses rolling cov/var for speed (beta only). Intercept handled in spread via mean-adjustment.
    """
    # beta = Cov(y,x) / Var(x)
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    return beta

def compute_spread_and_z(df: pd.DataFrame, lookback_beta: int, lookback_z: int) -> pd.DataFrame:
    out = df.copy()
    # log prices for stationarity-friendly spread
    out["ly"] = np.log(out["y"])
    out["lx"] = np.log(out["x"])

    out["beta"] = rolling_ols_beta(out["ly"], out["lx"], lookback_beta)

    # spread = ly - beta*lx  (intercept absorbed by rolling mean below)
    out["spread"] = out["ly"] - out["beta"] * out["lx"]

    m = out["spread"].rolling(lookback_z).mean()
    s = out["spread"].rolling(lookback_z).std(ddof=0)
    out["z"] = (out["spread"] - m) / s

    return out

def generate_positions(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """
    Position is +1 for long spread (long y, short x) and -1 for short spread.
    Entry when |z| >= entry_z, exit when |z| <= exit_z.
    """
    pos = pd.Series(index=z.index, dtype=float)
    state = 0.0

    for t, val in z.items():
        if np.isnan(val):
            pos.loc[t] = 0.0
            state = 0.0
            continue

        if state == 0.0:
            if val >= entry_z:
                state = -1.0  # short spread
            elif val <= -entry_z:
                state = +1.0  # long spread
        else:
            if abs(val) <= exit_z:
                state = 0.0

        pos.loc[t] = state

    return pos

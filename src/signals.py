# src/signals.py
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

def rolling_coint_pvalue(log_y: pd.Series, log_x: pd.Series, window: int) -> pd.Series:
    """
    Rolling Engle–Granger cointegration test p-values.
    H0: no cointegration. Lower p-value => more evidence of cointegration.
    """
    pvals = pd.Series(index=log_y.index, dtype=float)

    ly = log_y.values
    lx = log_x.values
    idx = log_y.index

    for i in range(window - 1, len(idx)):
        y_win = ly[i - window + 1 : i + 1]
        x_win = lx[i - window + 1 : i + 1]
        # coint returns (t_stat, p_value, crit_values)
        try:
            _, pval, _ = coint(y_win, x_win)
        except Exception:
            pval = np.nan
        pvals.iloc[i] = pval

    return pvals

def apply_coint_regime_filter(pos: pd.Series, pvals: pd.Series, p_threshold: float) -> pd.Series:
    """
    Force flat (0 position) when p-value indicates no cointegration.
    """
    tradable = (pvals < p_threshold)
    filtered = pos.where(tradable, 0.0)
    return filtered


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

def compute_spread_and_z(
    df: pd.DataFrame,
    lookback_beta: int,
    lookback_z: int,
    lookback_coint: int,
    coint_p_threshold: float,
) -> pd.DataFrame:
    """
    Computes:
      - log prices (ly, lx)
      - rolling hedge ratio beta (OLS proxy via rolling cov/var)
      - spread = ly - beta*lx
      - z-score of spread
      - rolling Engle–Granger cointegration p-values and boolean regime flag

    Parameters
    ----------
    df : DataFrame with columns ['y','x'] price series aligned by timestamp index
    lookback_beta : window length for rolling beta estimate
    lookback_z : window length for spread z-score mean/std
    lookback_coint : window length for rolling cointegration test
    coint_p_threshold : p-value threshold used to form is_coint flag

    Returns
    -------
    DataFrame with original cols plus: ly, lx, beta, spread, z, coint_p, is_coint
    """
    out = df.copy()

    # log prices for a more stable spread
    out["ly"] = np.log(out["y"])
    out["lx"] = np.log(out["x"])

    # rolling hedge ratio beta (fast OLS proxy)
    out["beta"] = rolling_ols_beta(out["ly"], out["lx"], lookback_beta)

    # spread definition (intercept absorbed by rolling mean)
    out["spread"] = out["ly"] - out["beta"] * out["lx"]

    # z-score of spread
    m = out["spread"].rolling(lookback_z).mean()
    s = out["spread"].rolling(lookback_z).std(ddof=0)
    out["z"] = (out["spread"] - m) / s

    # rolling cointegration p-values (regime detection)
    out["coint_p"] = rolling_coint_pvalue(out["ly"], out["lx"], lookback_coint)
    out["is_coint"] = out["coint_p"] < coint_p_threshold

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

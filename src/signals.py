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

def compute_vol_scale(spread: pd.Series, window: int, target_spread_vol: float,
                      min_scale: float, max_scale: float) -> pd.Series:
    """
    Volatility scaling based on rolling volatility of spread changes.
    scale_t = target_vol / rolling_std(diff(spread))
    """
    spread_d = spread.diff()
    vol = spread_d.rolling(window).std(ddof=0)

    scale = target_spread_vol / vol
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scale = scale.clip(lower=min_scale, upper=max_scale)
    return scale

def compute_spread_and_z(
    df: pd.DataFrame,
    lookback_beta: int,
    lookback_z: int,
    lookback_coint: int,
    coint_p_threshold: float,
    lookback_spread_vol: int,
    target_spread_vol: float,
    min_scale: float,
    max_scale: float,
) -> pd.DataFrame:
    out = df.copy()

    # log prices
    out["ly"] = np.log(out["y"])
    out["lx"] = np.log(out["x"])

    # rolling hedge ratio beta
    out["beta"] = rolling_ols_beta(out["ly"], out["lx"], lookback_beta)

    # spread
    out["spread"] = out["ly"] - out["beta"] * out["lx"]

    # z-score
    m = out["spread"].rolling(lookback_z).mean()
    s = out["spread"].rolling(lookback_z).std(ddof=0)
    out["z"] = (out["spread"] - m) / s

    # rolling cointegration p-values + regime flag
    out["coint_p"] = rolling_coint_pvalue(out["ly"], out["lx"], lookback_coint)
    out["is_coint"] = out["coint_p"] < coint_p_threshold

    # volatility scaling factor (risk targeting)
    out["vol_scale"] = compute_vol_scale(
        out["spread"],
        window=lookback_spread_vol,
        target_spread_vol=target_spread_vol,
        min_scale=min_scale,
        max_scale=max_scale,
    )

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
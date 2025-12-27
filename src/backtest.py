# src/backtest.py
import numpy as np
import pandas as pd

def backtest_pairs(
    df: pd.DataFrame,
    pos_spread: pd.Series,
    beta: pd.Series,
    vol_scale: pd.Series,
    gross_leverage: float,
    max_gross_leverage: float,
    fee_bps: float,
    slippage_bps: float,
    max_holding_bars: int,
) -> pd.DataFrame:

    """
    Trading spread:
      pos_spread = +1 => long y, short x * beta
      pos_spread = -1 => short y, long x * beta

    PnL approximation using log returns and beta hedge ratio.
    Costs applied on position changes (turnover), proportional to gross exposure.
    """

    out = df.copy()

    # log returns
    out["ry"] = np.log(out["y"]).diff()
    out["rx"] = np.log(out["x"]).diff()

    out["pos"] = pos_spread.fillna(0.0)
    out["beta"] = beta.ffill().fillna(0.0)
    out["vol_scale"] = vol_scale.reindex(out.index).ffill().fillna(0.0)


    # holding-period cap
    # if position held too long, force exit
    hold = 0
    pos_capped = []
    prev = 0.0
    for v in out["pos"].values:
        if v == 0.0:
            hold = 0
            pos_capped.append(0.0)
        else:
            if prev == v:
                hold += 1
            else:
                hold = 1
            if hold > max_holding_bars:
                pos_capped.append(0.0)
                hold = 0
            else:
                pos_capped.append(v)
        prev = pos_capped[-1]
    out["pos"] = pos_capped

    # translate spread position into leg weights (grossed to gross_leverage)
    # we target |w_y| + |w_x| = gross_leverage
    # w_y = pos * a ; w_x = -pos * a * beta
    # choose a so that abs(wy)+abs(wx)=gross
    # Effective gross exposure scales with spread volatility, capped
    eff_gross = gross_leverage * out["vol_scale"]
    eff_gross = eff_gross.clip(lower=0.0, upper=max_gross_leverage)

    denom = (1.0 + out["beta"].abs()).replace(0.0, np.nan)
    a = eff_gross / denom

    out["w_y"] = out["pos"] * a
    out["w_x"] = -out["pos"] * a * out["beta"]

    # portfolio return per bar (log-return approx)
    out["ret_gross"] = out["w_y"].shift(1) * out["ry"] + out["w_x"].shift(1) * out["rx"]
    out["ret_gross"] = out["ret_gross"].fillna(0.0)

    # costs on turnover
    cost_per_turn = (fee_bps + slippage_bps) / 1e4  # bps -> fraction
    turnover = (out["w_y"].diff().abs() + out["w_x"].diff().abs()).fillna(0.0)
    out["cost"] = cost_per_turn * turnover

    out["ret_net"] = out["ret_gross"] - out["cost"]
    out["equity"] = (1.0 + out["ret_net"]).cumprod()

    return out

# src/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    exchange_id: str = "kraken"
    symbol_y: str = "BTC/USD"
    symbol_x: str = "ETH/USD"
    timeframe: str = "1h"
    limit: int = 1500            # number of candles to pull per symbol

    # Strategy parameters
    lookback_beta: int = 200      # rolling window for hedge ratio estimate
    lookback_z: int = 200         # rolling window for z-score
    entry_z: float = 2.0
    exit_z: float = 0.5

    # Risk / sizing
    gross_leverage: float = 1.0   # total gross exposure (e.g., 1.0 => 100% gross)
    max_holding_bars: int = 24*14 # safety stop, ~2 weeks on 1h data

    # Costs (round-trip approx: 2 * fee + slippage). Tune per venue.
    fee_bps: float = 4.0          # e.g., 4 bps per side total? (adjust)
    slippage_bps: float = 2.0     # assumed slippage per side

    # Cointegration regime filter
    lookback_coint: int = 300
    coint_p_threshold: float = 0.05

    # Volatility scaling (risk targeting)
    lookback_spread_vol: int = 200  # window for spread volatility estimate
    target_spread_vol: float = 0.0015  # target spread-vol per bar (tune later)
    max_gross_leverage: float = 2.0  # cap effective gross exposure
    min_scale: float = 0.0  # allow 0
    max_scale: float = 3.0  # cap scaling multiplier


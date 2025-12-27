# Crypto Statistical Arbitrage: BTC–ETH Pairs Trading

## Overview
This project implements a market-neutral statistical arbitrage strategy on BTC and ETH
using rolling hedge ratios and z-score mean reversion. The goal is to evaluate whether
a naive pairs-trading approach produces risk-adjusted returns in crypto markets.

## Strategy Summary
- Hourly OHLCV data
- Log-price spread with rolling OLS hedge ratio
- Z-score based entry/exit rules
- Fixed gross leverage
- Transaction cost and slippage modeling

## Results (Baseline)
The naive implementation produces negative risk-adjusted returns over the tested window.
This highlights the instability of mean-reversion assumptions in trending crypto regimes.

## Key Takeaways
- BTC–ETH correlation does not imply persistent mean reversion
- Z-score signals fail during strong directional markets
- Risk controls prevent large drawdowns, but do not create edge

## Next Steps
- Rolling cointegration regime filter
- Volatility-scaled position sizing
- Funding-rate and execution modeling

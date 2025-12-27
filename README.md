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

## How to Run
- Run the following two commands to download dependencies and run the backtest 
- pip install -r requirements.txt 
- python src/run_backtest.py

## Results (Baseline)
The naive implementation produces negative risk-adjusted returns over the tested window.
This highlights the instability of mean-reversion assumptions in trending crypto regimes.

### Regime Filtering Result
Introducing a rolling Engle–Granger cointegration filter materially improved
risk-adjusted performance by preventing trades during non–mean-reverting regimes.

## Key Takeaways
- BTC–ETH correlation does not imply persistent mean reversion
- Z-score signals fail during strong directional markets
- Risk controls prevent large drawdowns, but do not create edge

## What This Project Demonstrates

- Construction of a market-neutral statistical arbitrage strategy using rolling hedge ratios
- Practical application of time-series statistics (log-price spreads, z-score normalization)
- End-to-end backtesting pipeline with transaction cost and slippage modeling
- Risk-aware evaluation using Sharpe ratio, drawdown, and hit rate
- Diagnosis of strategy failure modes under non–mean-reverting market regimes
- Clean, modular research code suitable for extension and iteration

## Next Steps
- Volatility-scaled position sizing
- Funding-rate and execution modeling

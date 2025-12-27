# src/run_backtest.py
import os
import json
import matplotlib.pyplot as plt

from config import Config
from data_loader import load_pair_close
from signals import compute_spread_and_z, generate_positions, apply_coint_regime_filter
from backtest import backtest_pairs
from metrics import summarize

from pathlib import Path


def main():
    cfg = Config()

    df = load_pair_close(cfg.exchange_id, cfg.symbol_y, cfg.symbol_x, cfg.timeframe, cfg.limit)
    feat = compute_spread_and_z(
        df,
        cfg.lookback_beta,
        cfg.lookback_z,
        cfg.lookback_coint,
        cfg.coint_p_threshold,
    )

    pos_raw = generate_positions(feat["z"], cfg.entry_z, cfg.exit_z)
    pos = apply_coint_regime_filter(pos_raw, feat["coint_p"], cfg.coint_p_threshold)

    out = backtest_pairs(
        df=df,
        pos_spread=pos,
        beta=feat["beta"],
        gross_leverage=cfg.gross_leverage,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        max_holding_bars=cfg.max_holding_bars,
    )

    stats = summarize(out, cfg.timeframe)
    print(json.dumps(stats, indent=2))

    os.makedirs("results", exist_ok=True)

    # --- Paths (robust to where script is run) ---
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    equity_path = RESULTS_DIR / "equity_curve.png"
    zscore_path = RESULTS_DIR / "zscore.png"

    # --- Equity Curve ---
    plt.figure(figsize=(10, 5))
    out["equity"].plot()
    plt.title(f"Equity Curve: {cfg.symbol_y} vs {cfg.symbol_x} ({cfg.timeframe})")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(equity_path, dpi=150)
    plt.close()

    # --- Z-Score Plot ---
    plt.figure(figsize=(10, 5))
    feat["z"].plot()
    plt.axhline(cfg.entry_z, linestyle="--", color="red", label="Entry")
    plt.axhline(-cfg.entry_z, linestyle="--", color="red")
    plt.axhline(cfg.exit_z, linestyle="--", color="green", label="Exit")
    plt.axhline(-cfg.exit_z, linestyle="--", color="green")
    plt.title("Spread Z-Score")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Z-Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(zscore_path, dpi=150)
    plt.close()

    print(f"Saved equity curve to: {equity_path}")
    print(f"Saved z-score plot to: {zscore_path}")

    # --- Cointegration p-value plot ---
    coint_path = RESULTS_DIR / "coint_pvalue.png"
    plt.figure(figsize=(10, 5))
    feat["coint_p"].plot()
    plt.axhline(cfg.coint_p_threshold, linestyle="--", label="p-threshold")
    plt.title("Rolling Engle–Granger Cointegration p-value")
    plt.xlabel("Time (UTC)")
    plt.ylabel("p-value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(coint_path, dpi=150)
    plt.close()
    print(f"Saved cointegration p-value plot to: {coint_path}")


if __name__ == "__main__":
    main()

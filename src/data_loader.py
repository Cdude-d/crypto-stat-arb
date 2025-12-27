# src/data_loader.py
import ccxt
import pandas as pd

def _fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    return df

def load_pair_close(exchange_id: str, symbol_y: str, symbol_x: str, timeframe: str, limit: int) -> pd.DataFrame:
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    y = _fetch_ohlcv_df(exchange, symbol_y, timeframe, limit)[["close"]].rename(columns={"close": "y"})
    x = _fetch_ohlcv_df(exchange, symbol_x, timeframe, limit)[["close"]].rename(columns={"close": "x"})

    df = y.join(x, how="inner").dropna()
    return df

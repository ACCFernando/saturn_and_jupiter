# src/data_fetch.py
import requests
import pandas as pd
from datetime import datetime
from typing import Optional

class DataFetcher:
    """
    A class for fetching cryptocurrency market data (candlesticks) from Binance.

    Attributes
    ----------
    base_url : str
        The base URL of the Binance API.
    symbol : str
        The trading pair symbol, e.g., "BTCUSDT".
    interval : str
        The candle interval, e.g., "1m", "5m".
    """

    def __init__(self, base_url: str, pair: str, interval: str):
        """
        Initialize a DataFetcher instance.

        Parameters
        ----------
        base_url : str
            Base URL of the exchange API.
        pair : str
            Trading pair symbol, e.g., "BTCUSDT".
        interval : str
            Candle interval, e.g., "1m".
        """
        self.base_url = base_url
        self.symbol = pair
        self.interval = interval

    def get_latest_klines(self, limit: int = 50) -> pd.DataFrame:
        """
        Fetch the latest candlestick (kline) data.

        Parameters
        ----------
        limit : int, optional
            Number of candles to fetch, by default 50.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing OHLCV data and timestamps.
            Columns: ["open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","trades",
            "taker_buy_base","taker_buy_quote","ignore"]

        Raises
        ------
        ValueError
            If the returned data is not in the expected format.
        """
        endpoint = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit
        }
        response = requests.get(endpoint, params=params)
        data = response.json()

        if not isinstance(data, list):
            raise ValueError("Unexpected response format from Binance API.")

        columns = ["open_time","open","high","low","close","volume",
                   "close_time","quote_asset_volume","trades",
                   "taker_buy_base","taker_buy_quote","ignore"]

        df = pd.DataFrame(data, columns=columns)
        df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
        df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')

        numeric_cols = ["open","high","low","close","volume",
                        "quote_asset_volume","taker_buy_base","taker_buy_quote"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)

        return df

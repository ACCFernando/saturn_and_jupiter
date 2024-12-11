# src/strategy.py
import pandas as pd
import numpy as np
from src.ml_model import MLModel

class SimpleMA_Strategy:
    def __init__(self, fast_period: int = 3, slow_period: int = 8):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> str:
        df["fast_ma"] = df["close"].rolling(self.fast_period).mean()
        df["slow_ma"] = df["close"].rolling(self.slow_period).mean()

        fast_val = df["fast_ma"].iloc[-1]
        slow_val = df["slow_ma"].iloc[-1]

        if fast_val > slow_val:
            return "BUY"
        elif fast_val < slow_val:
            return "SELL"
        else:
            return "HOLD"


class AdvancedSignalStrategy:
    def __init__(self, 
                 bb_period: int = 20, 
                 bb_stddev: float = 2.0, 
                 rsi_period: int = 14, 
                 rsi_overbought: int = 70, 
                 rsi_oversold: int = 30):
        self.bb_period = bb_period
        self.bb_stddev = bb_stddev
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def generate_signals(self, df: pd.DataFrame) -> str:
        df["ma"] = df["close"].rolling(self.bb_period).mean()
        df["std"] = df["close"].rolling(self.bb_period).std()
        df["upper_bb"] = df["ma"] + (self.bb_stddev * df["std"])
        df["lower_bb"] = df["ma"] - (self.bb_stddev * df["std"])

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        current_close = df["close"].iloc[-1]
        upper_bb = df["upper_bb"].iloc[-1]
        lower_bb = df["lower_bb"].iloc[-1]
        rsi_val = df["rsi"].iloc[-1]

        if (current_close <= lower_bb) and (rsi_val < self.rsi_oversold):
            return "BUY"
        elif (current_close >= upper_bb) and (rsi_val > self.rsi_overbought):
            return "SELL"
        else:
            return "HOLD"


class HybridMLStrategy:
    """
    Combines AdvancedSignalStrategy and MLModel predictions.
    The final signal might be:
    - If ML says BUY and advanced says BUY => BUY
    - If ML says SELL and advanced says SELL => SELL
    - Else HOLD
    """
    def __init__(self):
        self.adv_strategy = AdvancedSignalStrategy()
        self.ml_model = MLModel()

    def generate_signals(self, df: pd.DataFrame) -> str:
        adv_signal = self.adv_strategy.generate_signals(df)
        ml_signal = self.ml_model.predict_signal(df)

        # Simple consensus logic
        if adv_signal == ml_signal:
            return adv_signal
        else:
            # If disagree, we can choose to HOLD or trust one more than the other
            return "HOLD"

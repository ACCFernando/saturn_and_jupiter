# src/backtest.py
import pandas as pd
from typing import Tuple
from src.paper_trader import PaperTrader
from src.strategy import HybridMLStrategy

class Backtester:
    def __init__(self, take_profit_pct: float = 0.001, stop_loss_pct: float = 0.001, position_size_usd: float = 100):
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self.position_size_usd = position_size_usd
        self.strategy = HybridMLStrategy()  # Use the hybrid strategy

    def run_backtest(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        df = df.sort_values(by='open_time').reset_index(drop=True)

        trades = []
        position = None
        balance = 0.0
        quantity = None

        # Need enough data for indicators
        min_bars = 20
        for i in range(min_bars, len(df)):
            # Use .copy() to ensure we have a proper DataFrame with column names
            signal_df = df.iloc[:i+1].copy()  
            signal = self.strategy.generate_signals(signal_df)
            last_price = df["close"].iloc[i]
            current_time = df["open_time"].iloc[i]

            if position is None:
                if signal in ["BUY", "SELL"]:
                    position = {
                        "side": signal,
                        "entry_price": last_price,
                        "entry_time": current_time
                    }
                    quantity = self.position_size_usd / last_price
            else:
                side = position["side"]
                entry_price = position["entry_price"]
                # Check TP/SL
                if side == "BUY":
                    if (last_price >= entry_price*(1+self.take_profit_pct)) or (last_price <= entry_price*(1-self.stop_loss_pct)):
                        pnl = (last_price - entry_price)*quantity
                        balance += pnl
                        trades.append({
                            "entry_time": position["entry_time"],
                            "entry_price": entry_price,
                            "exit_time": current_time,
                            "exit_price": last_price,
                            "side": side,
                            "pnl_usd": pnl
                        })
                        position = None
                        quantity = None
                else: # SELL
                    if (last_price <= entry_price*(1-self.take_profit_pct)) or (last_price >= entry_price*(1+self.take_profit_pct)):
                        pnl = (entry_price - last_price)*quantity
                        balance += pnl
                        trades.append({
                            "entry_time": position["entry_time"],
                            "entry_price": entry_price,
                            "exit_time": current_time,
                            "exit_price": last_price,
                            "side": side,
                            "pnl_usd": pnl
                        })
                        position = None
                        quantity = None

        trades_df = pd.DataFrame(trades)
        return trades_df, balance

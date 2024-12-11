from .config import Config
from .data_fetch import DataFetcher
from .strategy import SimpleMA_Strategy
from .execution import ExecutionManager
from .paper_trader import PaperTrader
import time

def main(run_once=False):
    config = Config("configs/config_example.json")
    fetcher = DataFetcher(config.base_url, config.trading_pair, config.interval)
    strategy = SimpleMA_Strategy(config.fast_ma_period, config.slow_ma_period)
    executor = ExecutionManager(paper_trading=config.paper_trading)
    trader = PaperTrader(config.starting_balance)

    print(f"Starting scalping bot on {config.trading_pair} with ${config.starting_balance} initial balance.")

    while True:
        try:
            df = fetcher.get_latest_klines(limit=50)
            if len(df) < max(config.fast_ma_period, config.slow_ma_period):
                print("Not enough data yet.")
                if run_once:
                    break
                time.sleep(60)
                continue

            signal = strategy.generate_signals(df)
            last_price = df["close"].iloc[-1]

            position = trader.get_position()
            if position is None:
                if signal == "BUY":
                    executor.place_order("BUY", config.position_size_usd, last_price)
                    trader.open_position("BUY", config.position_size_usd, last_price)
                    print(f"Opened BUY at {last_price}, balance: {trader.get_balance()}")
                elif signal == "SELL":
                    executor.place_order("SELL", config.position_size_usd, last_price)
                    trader.open_position("SELL", config.position_size_usd, last_price)
                    print(f"Opened SELL at {last_price}, balance: {trader.get_balance()}")
            else:
                side = position["side"]
                entry_price = position["entry_price"]
                if side == "BUY":
                    if last_price >= entry_price * (1 + config.take_profit_pct) or \
                       last_price <= entry_price * (1 - config.stop_loss_pct):
                        pnl = trader.close_position(last_price)
                        print(f"Closed BUY at {last_price}, PnL: {pnl}, New Balance: {trader.get_balance()}")

                elif side == "SELL":
                    if last_price <= entry_price * (1 - config.take_profit_pct) or \
                       last_price >= entry_price * (1 + config.stop_loss_pct):
                        pnl = trader.close_position(last_price)
                        print(f"Closed SELL at {last_price}, PnL: {pnl}, New Balance: {trader.get_balance()}")

            if run_once:
                break

            time.sleep(60)
        except Exception as e:
            print(f"Error: {e}")
            if run_once:
                break
            time.sleep(60)

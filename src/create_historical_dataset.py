# src/create_historical_dataset.py
import sys
import os
import pandas as pd

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Config
from src.data_fetch import DataFetcher

# Load configuration
config = Config("configs/config_example.json")

# Initialize DataFetcher with parameters from config
fetcher = DataFetcher(
    base_url=config.base_url,
    pair=config.trading_pair,
    interval=config.interval
)

# Fetch the latest 1000 candles (adjust as needed if the API supports that many)
df = fetcher.get_latest_klines(limit=1000)

# Save the retrieved DataFrame to CSV
output_path = "data/historical_data.csv"
df.to_csv(output_path, index=False)
print(f"Historical data saved to {output_path}")
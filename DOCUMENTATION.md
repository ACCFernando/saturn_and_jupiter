# Project Documentation

This documentation provides a detailed description of the variables, data processing steps, and financial techniques used in the crypto trading AI system. It covers data inputs, feature engineering, model training, backtesting, evaluation, and considerations for live trading deployment.

## Overview

The system is designed to predict short-term price movements of a cryptocurrency trading pair (e.g., BTC/USDT) and execute trades accordingly. The pipeline includes:

1. **Data Fetching**: Retrieve historical and live market data (OHLCV - open, high, low, close, volume) from an exchange.
2. **Feature Engineering**: Compute technical indicators such as moving averages, Bollinger Bands, and RSI.
3. **Target Definition**: Predict whether the next candle’s closing price will be higher than the current one.
4. **Model Training & Evaluation**: Train an ML model, validate it through backtests, measure performance via metrics like AUC, confusion matrix, and KS statistic.
5. **Execution Logic**: Use signals from the model to issue buy/sell commands, integrate risk management rules, and eventually deploy in a live environment.

---

## Variables and Parameters

### Data and Configuration Variables

- **`config.base_url`**: Base endpoint for API requests (e.g., `https://api.binance.com`).
- **`config.trading_pair`**: The symbol of the trading pair (e.g., `BTCUSDT`).
- **`config.interval`**: The candle interval (e.g., `"1m"`, `"5m"`, `"1h"`), defining the period each candle represents.
- **`config.starting_balance`**: Initial balance in USD (or another currency) for paper trading or simulation.
- **`config.take_profit_pct`, `config.stop_loss_pct`**: Percentage thresholds for automatic trade exits at profits or losses.
- **`config.position_size_usd`**: The notional amount of USD to allocate per trade.

### Data Fields in Historical Data (OHLCV)

- **`open_time`**: The timestamp of when the candle opened.
- **`open`, `high`, `low`, `close`**: Prices at candle open, highest during the period, lowest during the period, and close at the end.
- **`volume`**: Total trading volume during the candle’s period.

---

## Feature Engineering Variables

These variables are computed from the `close` price series:

- **`fast_ma` (Fast Moving Average)**:  
  A short-period moving average (e.g., 3-period) that reacts quickly to recent price changes.
- **`slow_ma` (Slow Moving Average)**:  
  A longer-period moving average (e.g., 8-period) that smooths out price fluctuations more aggressively.
- **`ma` (Moving Average for Bollinger Bands)**:  
  Typically a 20-period moving average used as the center line of Bollinger Bands.
- **`std` (Standard Deviation)**:  
  The standard deviation of the closing price over the chosen period (often 20). Used to measure volatility.
- **`upper_bb`, `lower_bb` (Bollinger Bands)**:  
  Computed as `ma ± (std_dev_factor * std)`. Commonly, `std_dev_factor = 2.0`.  
  These bands form dynamic support/resistance levels.
- **`rsi` (Relative Strength Index)**:  
  A momentum oscillator measuring recent price gains vs. losses over a defined period (commonly 14).  
  RSI values range from 0 to 100:
  - Below 30 often considered oversold.
  - Above 70 considered overbought.

**How They Are Used:**
- Moving averages and Bollinger Bands help identify trend direction and volatility.
- RSI identifies potential overbought/oversold conditions, hinting at price reversals.

---

## Target Variable

- **`target`**: A binary label indicating whether the **next** candle’s closing price is higher than the current one.  
  - `target = 1` if `future_close > current_close`
  - `target = 0` otherwise

**Interpretation:** The model predicts short-term price direction on the next period (defined by the candle interval).

---

## Model Training Variables & Techniques

- **Features**: `[fast_ma, slow_ma, upper_bb, lower_bb, rsi]`  
  Input into the machine learning classifier (e.g., Gradient Boosting Classifier).
- **Train/Test Split**:  
  The dataset is split into training and test subsets (e.g., 80/20) to evaluate generalization.
- **Classification Goal**:  
  Predict `target` using extracted features. This is a supervised classification problem.
- **Gradient Boosting Classifier**:  
  An ensemble method that builds an additive model in a stage-wise fashion, focusing on errors of previous models to improve predictions.

---

## Model Evaluation Metrics

- **Accuracy**: Fraction of correct predictions.
- **Precision, Recall, F1-score**:  
  - **Precision**: Among predicted positives, how many are correct?
  - **Recall**: Among actual positives, how many were correctly identified?
  - **F1-score**: Harmonic mean of precision and recall.
- **AUC (Area Under ROC Curve)**:  
  Measures how well the model ranks positive cases higher than negative ones; 1.0 is perfect.
- **ROC Curve (Receiver Operating Characteristic)**:  
  Plots the true positive rate vs. false positive rate at various thresholds.
- **Confusion Matrix**:  
  A table showing counts of true positives, false positives, true negatives, and false negatives.
- **KS Statistic (Kolmogorov-Smirnov)**:  
  Measures the maximum difference between cumulative distributions of predicted probabilities for positives and negatives.
- **KS p-value**:  
  Indicates if the observed separation (KS statistic) is statistically significant or due to chance.

---

## Backtesting Variables

- **`backtester.run_backtest(df)`**:  
  Uses historical data `df` to simulate trades based on model predictions.
- **Trade Execution in Backtest**:  
  Signals ("BUY", "SELL") trigger simulated trades, recording entry and exit prices, and computing PnL (Profit & Loss).

**Objective:** Validate strategy performance on past data before live trading.

---

## Risk Management & Trading Parameters

- **Stop-Loss**: Automatic exit if price moves against the position beyond a certain percentage.
- **Take-Profit**: Automatic exit upon reaching a predefined profit target.
- **Position Sizing**: Determined by `position_size_usd`, controlling how much capital to commit per trade.
- **Daily/Session Limits**: Can be implemented to halt trading after a certain loss threshold.

**Rationale:** Ensures controlled losses and preserves capital.

---

## Live Trading Considerations

- **API Keys & Authentication**:  
  Required for live order execution on real exchanges.
- **Real-Time Data Streams (e.g., WebSockets)**:  
  Continuously fetch updated prices, recompute features, and generate new signals.
- **Execution Manager**:  
  Sends live buy/sell orders, handles order statuses, and errors.
- **Monitoring & Alerts**:  
  Use dashboards, logs, and alerts for system health and unexpected events.

---

## Financial Techniques Summary

1. **OHLCV Candlestick Data**: Standard time-series format in trading.
2. **Moving Averages**: Identify trends.
3. **Bollinger Bands**: Gauge volatility and potential mean reversion points.
4. **RSI**: Momentum indicator for overbought/oversold conditions.
5. **Binary Direction Prediction**: Predict whether price will rise or fall next period.
6. **Probabilistic Model Outputs**: Models provide probability estimates for upward moves.
7. **Risk Management Tools (Stop-Loss/Take-Profit)**: Core techniques to limit losses and secure profits.

---

## Conclusion

This documentation explains each variable, parameter, and financial technique used in the system. It outlines how data is processed, how features are engineered, how the model is trained and evaluated, and how to integrate the model’s predictions into a live trading environment with robust risk management and monitoring.

Save this file as `DOCUMENTATION.md` in the project’s root directory.

# src/ml_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

class MLModel:
    """
    Loads a trained ML model to predict short-term price direction.
    If model not found, can train from historical_data.csv.
    """

    def __init__(self, model_path: str = "models/trained_model.pkl"):
        self.model_path = model_path
        self.model = None

    def train_model(self, df: pd.DataFrame):
        """
        Train a simple model on historical data.
        Label: 1 if next close > current close else 0.
        Features: RSI, Bollinger Bands, MAs
        """
        df = df.copy().dropna()

        # Simple labeling: next_close > current_close?
        df["future_close"] = df["close"].shift(-1)
        df["target"] = (df["future_close"] > df["close"]).astype(int)
        df.dropna(inplace=True)

        # Features (example set)
        # Already computed by advanced strategy logic or we replicate here:
        df["fast_ma"] = df["close"].rolling(3).mean()
        df["slow_ma"] = df["close"].rolling(8).mean()
        df["ma"] = df["close"].rolling(20).mean()
        df["std"] = df["close"].rolling(20).std()
        df["upper_bb"] = df["ma"] + 2*df["std"]
        df["lower_bb"] = df["ma"] - 2*df["std"]

        # RSI Calculation (same as in strategy)
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)

        features = ["fast_ma","slow_ma","upper_bb","lower_bb","rsi"]
        X = df[features]
        y = df["target"]

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        self.model = model

        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)
        print(f"Model trained and saved to {self.model_path}")

    def load_model(self):
        """
        Load the trained model from disk.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully.")
        else:
            raise FileNotFoundError("Trained model not found. Train the model first.")

    def predict_signal(self, df: pd.DataFrame) -> str:
        """
        Predict next candle direction given current state.
        Returns: "BUY" if model predicts upward move, "SELL" if downward,
        else "HOLD" if unsure.
        """
        if self.model is None:
            self.load_model()

        # Compute same features on the last row
        if len(df) < 20:
            return "HOLD"  # Not enough data for indicators

        temp = df.copy().dropna()
        # Recompute indicators for last row
        temp["fast_ma"] = temp["close"].rolling(3).mean()
        temp["slow_ma"] = temp["close"].rolling(8).mean()
        temp["ma"] = temp["close"].rolling(20).mean()
        temp["std"] = temp["close"].rolling(20).std()
        temp["upper_bb"] = temp["ma"] + 2*temp["std"]
        temp["lower_bb"] = temp["ma"] - 2*temp["std"]

        delta = temp["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        temp["rsi"] = 100 - (100 / (1 + rs))

        temp.dropna(inplace=True)
        if len(temp) == 0:
            return "HOLD"
        
        last_row = temp.iloc[-1]
        X = [last_row[["fast_ma","slow_ma","upper_bb","lower_bb","rsi"]].values]

        pred = self.model.predict(X)[0]
        # If pred = 1 => upward move expected => "BUY"
        # If pred = 0 => downward move => "SELL"
        # Optionally add confidence threshold here
        return "BUY" if pred == 1 else "SELL"

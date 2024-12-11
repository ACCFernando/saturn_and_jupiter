# src/model_evaluation.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split, learning_curve
from scipy.stats import ks_2samp
import joblib
from jinja2 import Template

class ModelEvaluation:
    """
    A class to evaluate the trained ML model on a test dataset.
    It generates:
    - Accuracy, Balanced Accuracy, Precision, Recall, F1, AUC, KS statistic
    - Confusion matrix, ROC curve, and learning curve plots
    - An HTML report summarizing all results
    """

    def __init__(self, 
                 model_path="models/trained_model.pkl", 
                 data_path="data/historical_data.csv",
                 output_dir="model_performance"):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self):
        df = pd.read_csv(self.data_path)
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"])

        # Feature engineering as per the training setup
        df["fast_ma"] = df["close"].rolling(3).mean()
        df["slow_ma"] = df["close"].rolling(8).mean()
        df["ma"] = df["close"].rolling(20).mean()
        df["std"] = df["close"].rolling(20).std()
        df["upper_bb"] = df["ma"] + 2*df["std"]
        df["lower_bb"] = df["ma"] - 2*df["std"]

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        df.dropna(inplace=True)

        df["future_close"] = df["close"].shift(-1)
        df["target"] = (df["future_close"] > df["close"]).astype(int)
        df.dropna(inplace=True)

        features = ["fast_ma","slow_ma","upper_bb","lower_bb","rsi"]
        X = df[features]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        model = joblib.load(self.model_path)
        return model

    def plot_roc_curve(self, y_test, y_proba):
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        plt.plot([0,1],[0,1],'r--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        roc_path = os.path.join(self.output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=100)
        plt.close()
        return auc_score, roc_path

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=100)
        plt.close()
        return cm_path

    def plot_learning_curve(self, model, X_train, y_train):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1,
            train_sizes=np.linspace(0.1,1.0,5)
        )

        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(6,6))
        plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        plt.title('Learning Curve')
        plt.xlabel('Training examples')
        plt.ylabel('Score')
        plt.legend(loc='best')
        lc_path = os.path.join(self.output_dir, "learning_curve.png")
        plt.savefig(lc_path, dpi=100)
        plt.close()
        return lc_path

    def compute_ks_statistic(self, y_test, y_proba):
        pos_proba = y_proba[y_test == 1]
        neg_proba = y_proba[y_test == 0]
        ks_stat, p_value = ks_2samp(pos_proba, neg_proba)
        return ks_stat, p_value

    def generate_report(self, metrics, roc_path, cm_path, lc_path):
        template_str = """
        <html>
        <head><title>Model Performance Report</title></head>
        <body>
          <h1>Model Performance Report</h1>
          <h2>Metrics</h2>
          <table border="1" cellspacing="0" cellpadding="5">
            <tr><th>Metric</th><th>Value</th></tr>
            {% for k,v in metrics.items() %}
            <tr><td>{{k}}</td><td>{{v}}</td></tr>
            {% endfor %}
          </table>

          <h2>ROC Curve</h2>
          <img src="{{roc_path}}" alt="ROC Curve" width="400">

          <h2>Confusion Matrix</h2>
          <img src="{{cm_path}}" alt="Confusion Matrix" width="400">

          <h2>Learning Curve</h2>
          <img src="{{lc_path}}" alt="Learning Curve" width="400">

          <h2>Classification Report</h2>
          <pre>{{classification_report}}</pre>
        </body>
        </html>
        """
        template = Template(template_str)
        report_html = template.render(
            metrics=metrics,
            roc_path=os.path.basename(roc_path),
            cm_path=os.path.basename(cm_path),
            lc_path=os.path.basename(lc_path),
            classification_report=metrics["Classification Report"]
        )

        report_path = os.path.join(self.output_dir, "report.html")
        with open(report_path, "w") as f:
            f.write(report_html)
        print(f"Report saved to {report_path}")

    def run_evaluation(self):
        X_train, X_test, y_train, y_test = self.load_data()
        model = self.load_model()

        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:,1]
        else:
            y_proba = y_pred.astype(float)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc_score, roc_path = self.plot_roc_curve(y_test, y_proba)
        cm_path = self.plot_confusion_matrix(y_test, y_pred)
        lc_path = self.plot_learning_curve(model, X_train, y_train)
        ks_stat, p_value = self.compute_ks_statistic(y_test, y_proba)

        cls_report = classification_report(y_test, y_pred)

        metrics = {
            "Accuracy": round(acc, 4),
            "Balanced Accuracy": round(balanced_acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-score": round(f1, 4),
            "AUC": round(auc_score, 4),
            "KS Statistic": round(ks_stat, 4),
            "KS p-value": p_value,
            "Classification Report": cls_report
        }

        self.generate_report(metrics, roc_path, cm_path, lc_path)


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.run_evaluation()

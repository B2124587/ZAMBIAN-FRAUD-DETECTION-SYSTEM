"""
MODULE 3: Machine Learning Models & Interface
Mobile Money Fraud Detection System - Zambia
models.py

Abstract MLModel interface + concrete implementations:
  - RandomForestFraudModel
  - XGBoostFraudModel
  - LogisticRegressionFraudModel
  - SmishingNLPClassifier (TF-IDF + RandomForest / NaiveBayes)
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# ABSTRACT BASE CLASS
# ─────────────────────────────────────────────

class MLModel(ABC):
    """Abstract interface for all fraud detection classifiers."""

    name: str = "BaseModel"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "MLModel":
        """Train the model on the provided dataset."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities. Shape: (n_samples, 2)."""
        ...

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions from probability threshold."""
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
        """Return a comprehensive evaluation dictionary."""
        y_proba = self.predict_proba(X)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        return {
            "model": self.name,
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(y, y_proba), 4) if len(np.unique(y)) > 1 else 0.0,
            "mcc": round(matthews_corrcoef(y, y_pred), 4),
        }

    def save(self, path: Path | None = None):
        p = path or MODEL_DIR / f"{self.name.lower().replace(' ', '_')}.pkl"
        with open(p, "wb") as f:
            pickle.dump(self, f)
        print(f"[models] Saved {self.name} → {p}")

    @classmethod
    def load(cls, path: Path) -> "MLModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# CONCRETE IMPLEMENTATIONS
# ─────────────────────────────────────────────

class RandomForestFraudModel(MLModel):
    """Primary classifier: Random Forest with calibrated class weights."""

    name = "RandomForest"

    def __init__(self):
        self._model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestFraudModel":
        print(f"[models] Training {self.name}...")
        self._model.fit(X_train, y_train)
        print(f"[models] {self.name} training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_


class XGBoostFraudModel(MLModel):
    """Secondary classifier: XGBoost gradient boosting."""

    name = "XGBoost"

    def __init__(self):
        self._model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=39,   # ~1/0.025 to handle class imbalance
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostFraudModel":
        print(f"[models] Training {self.name}...")
        self._model.fit(X_train, y_train)
        print(f"[models] {self.name} training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)


class LogisticRegressionFraudModel(MLModel):
    """Baseline classifier: Logistic Regression with feature scaling."""

    name = "LogisticRegression"

    def __init__(self):
        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionFraudModel":
        print(f"[models] Training {self.name}...")
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y_train)
        print(f"[models] {self.name} training complete.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)


# ─────────────────────────────────────────────
# NLP SMISHING CLASSIFIER
# ─────────────────────────────────────────────

SMISHING_CORPUS = [
    # Positive (smishing) examples
    ("Congratulations! You have won K5000. Click http://bit.ly/claim to receive your prize.", 1),
    ("URGENT: Your MTN account will be suspended. Verify now: mtn-zambia-verify.com", 1),
    ("Dear customer, your SIM needs immediate verification. Reply with PIN to 1234.", 1),
    ("Win a FREE iPhone! Airtel lottery selected you. Visit airtel-prize.tk now!", 1),
    ("Your Airtel Money account is locked. Send OTP to 0977123456 immediately.", 1),
    ("Bank alert: unusual login detected. Confirm identity via link: bit.ly/zamlk", 1),
    ("MTN: You have a pending K2,500 transfer. Confirm by sending your PIN to 777.", 1),
    ("Your Zanaco account suspended. Verify at zambia-bank-secure.ml urgently.", 1),
    ("WINNER! Zamtel prize draw: K10,000 cash. Claim at zamtel-winner.cf today.", 1),
    ("Security alert: 3 failed login attempts on your account. Tap to secure now.", 1),
    ("Free bonus K200 credited! Claim before midnight: airtelmoney-bonus.gq", 1),
    ("Your mobile money PIN has been compromised. Reset via link: secure-zm.tk", 1),
    ("ZESCO bill overdue. Pay now or face disconnection. Click: zesco-pay.ml", 1),
    ("Your parcel from DHL is held. Pay K50 clearance: dhl-zambia-delivery.com", 1),
    ("Govt COVID relief fund K3000 available for you. Register: zm-relief.tk", 1),
    # Negative (legitimate) examples
    ("Your MTN Mobile Money transfer of K150 to 0976543210 was successful.", 0),
    ("Airtel: Your account balance is K320.50. Dial *115# to check transactions.", 0),
    ("Zamtel: Your monthly bundle expires in 3 days. Renew by dialing *171#.", 0),
    ("Transaction confirmed: K500 sent to John M. Ref: TXN20240601. Thank you.", 0),
    ("Your Airtel Money PIN was changed successfully. Contact 333 if not you.", 0),
    ("MTN: Your data bundle of 1GB has been activated. Valid for 30 days.", 0),
    ("Thank you for paying your ZESCO bill of K180. Receipt: ZE20240512.", 0),
    ("Zanaco: Your account statement is ready. Login at zanaco.co.zm to view.", 0),
    ("Your Zamtel SIM card registration is complete. Welcome to Zamtel network.", 0),
    ("Reminder: Your loan repayment of K250 is due on 15th. Pay via *321#.", 0),
    ("MTN Mobile Money: Cash-in of K1,000 from Agent 0760001234 confirmed.", 0),
    ("Airtel: Low balance alert. Your account has K12.30. Recharge to stay connected.", 0),
    ("Your salary of K8,500 has been credited to your account. - Your Bank", 0),
    ("School fees payment of K2,200 to St. Mary's confirmed. Ref: SCH004412.", 0),
    ("Zamtel broadband bill K350 due 30 June. Pay via ZamPay or *171#.", 0),
]


class SmishingNLPClassifier(MLModel):
    """
    NLP classifier for detecting smishing SMS messages.
    Uses TF-IDF vectorisation + Multinomial Naive Bayes (primary)
    with a Random Forest alternative.
    """

    name = "SmishingNLP"

    def __init__(self, use_naive_bayes: bool = True):
        self.use_naive_bayes = use_naive_bayes
        if use_naive_bayes:
            self._pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                )),
                ("clf", MultinomialNB(alpha=0.1)),
            ])
        else:
            self._pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                )),
                ("clf", RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced",
                )),
            ])
        self._trained = False

    def _get_corpus(self) -> tuple[list[str], list[int]]:
        texts, labels = zip(*SMISHING_CORPUS)
        return list(texts), list(labels)

    def fit(self, X_train=None, y_train=None) -> "SmishingNLPClassifier":
        """
        Trains on built-in corpus (+ optional external data).
        X_train can be list of strings; y_train list of 0/1.
        """
        texts, labels = self._get_corpus()
        if X_train is not None and y_train is not None:
            texts.extend(list(X_train))
            labels.extend(list(y_train))
        print(f"[models] Training {self.name} on {len(texts)} samples...")
        self._pipeline.fit(texts, labels)
        self._trained = True
        print(f"[models] {self.name} training complete.")
        return self

    def predict_proba(self, X) -> np.ndarray:
        """X: list of strings (SMS text messages)."""
        if not self._trained:
            self.fit()
        return self._pipeline.predict_proba(X)

    def is_smishing(self, sms_text: str, threshold: float = 0.5) -> bool:
        proba = self.predict_proba([sms_text])[0, 1]
        return proba >= threshold

    # Override evaluate for text data
    def evaluate_text(self, texts: list[str], labels: list[int]) -> dict:
        y_proba = self.predict_proba(texts)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        y = np.array(labels)
        return {
            "model": self.name,
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(y, y_proba), 4),
            "mcc": round(matthews_corrcoef(y, y_pred), 4),
        }


# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────

def get_all_models() -> list[MLModel]:
    return [
        RandomForestFraudModel(),
        XGBoostFraudModel(),
        LogisticRegressionFraudModel(),
    ]


if __name__ == "__main__":
    # Quick smoke-test
    X_dummy = np.random.rand(100, 11).astype(np.float32)
    y_dummy = np.random.randint(0, 2, 100)

    for m in get_all_models():
        m.fit(X_dummy, y_dummy)
        metrics = m.evaluate(X_dummy, y_dummy)
        print(metrics)

    smishing = SmishingNLPClassifier()
    smishing.fit()
    print(smishing.is_smishing("Congratulations! You have won K5000. Claim now: bit.ly/zamwin"))
    print(smishing.is_smishing("Your balance is K320. Dial *115# to check."))

"""The five model families of Section 4.8 and their preprocessing.

Linear models and the network receive log(1 + x) of the heavy-tailed non-negative features
followed by standardisation, both estimated on the rows they are fitted on (never on
validation or test rows). Tree ensembles receive the features unchanged. Every family is
weighted inversely to class frequency; the network through sample weights.
"""
import os

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score
from sklearn.utils.fixes import parse_version

SCALED = {"logreg", "svm", "ffnn"}
DEFAULT_THRESHOLD = {"logreg": 0.5, "svm": 0.0, "rf": 0.5, "hgb": 0.5, "ffnn": 0.5}
LOG_EXACT = {"amount_paid_base", "amount_received_base"}
LOG_SUFFIXES = ("_cnt", "_sum", "_n_cp", "_hours_since_prev", "_in_deg", "_out_deg",
                "_two_hop", "_pagerank", "_component_size")
_SK18 = parse_version(sklearn.__version__) >= parse_version("1.8")


def log_columns(names) -> list:
    return [i for i, c in enumerate(names) if c in LOG_EXACT or c.endswith(LOG_SUFFIXES)]


class Preprocessor:
    """log(1 + x) on heavy-tailed columns, then standardisation; fitted on training rows."""

    def __init__(self, names):
        self.names, self.idx = list(names), log_columns(names)

    def _log(self, X):
        Z = np.array(X, dtype=np.float32, copy=True)
        if self.idx:
            Z[:, self.idx] = np.log1p(np.maximum(Z[:, self.idx], 0.0))
        return Z

    def fit(self, X):
        Z = self._log(X)
        self.mean_ = Z.mean(axis=0, dtype=np.float64).astype(np.float32)
        sd = Z.std(axis=0, dtype=np.float64)
        self.scale_ = np.where(sd > 0, sd, 1.0).astype(np.float32)
        return self

    def transform(self, X):
        Z = self._log(X)
        Z -= self.mean_
        Z /= self.scale_
        return Z


class KerasMLP:
    """Feed-forward network of one or two hidden layers (Keras), with balanced sample
    weights and early stopping on validation average precision."""

    def __init__(self, hidden_layers=1, units=64, dropout=0.0, learning_rate=1e-3,
                 batch_size=2048, max_epochs=20, patience=3, class_weight="balanced",
                 random_state=0):
        self.hidden_layers, self.units, self.dropout = int(hidden_layers), int(units), float(dropout)
        self.learning_rate, self.batch_size = float(learning_rate), int(batch_size)
        self.max_epochs, self.patience = int(max_epochs), int(patience)
        self.class_weight, self.random_state = class_weight, int(random_state)

    def fit(self, X, y, X_val=None, y_val=None):
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")
        import keras
        keras.utils.set_random_seed(self.random_state)
        layers = [keras.Input(shape=(X.shape[1],))]
        for _ in range(self.hidden_layers):
            layers.append(keras.layers.Dense(self.units, activation="relu",
                                             kernel_initializer="he_uniform"))
            if self.dropout > 0:
                layers.append(keras.layers.Dropout(self.dropout))
        layers.append(keras.layers.Dense(1, activation="sigmoid"))
        model = keras.Sequential(layers)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss="binary_crossentropy")
        y = np.asarray(y, dtype=np.float32)
        sw = None
        if self.class_weight == "balanced":
            n, n_pos = len(y), float(y.sum())
            sw = np.where(y == 1, n / (2 * n_pos), n / (2 * (n - n_pos))).astype(np.float32)
        best_ap, best_w, wait = -1.0, None, 0
        for epoch in range(self.max_epochs):
            model.fit(X, y, sample_weight=sw, batch_size=self.batch_size, epochs=1,
                      verbose=0, shuffle=True)
            self.epochs_ = epoch + 1
            if X_val is None:
                continue
            ap = average_precision_score(y_val, model.predict(X_val, batch_size=65536,
                                                              verbose=0).ravel())
            if ap > best_ap:
                best_ap, best_w, wait = ap, model.get_weights(), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_w is not None:
            model.set_weights(best_w)
        self.model_, self.best_val_ap_ = model, best_ap
        return self

    def decision_function(self, X):
        return self.model_.predict(X, batch_size=65536, verbose=0).ravel()


def make(family: str, params: dict, seed: int, class_weight="balanced"):
    p = dict(params)
    if family == "logreg":                  # penalised logistic regression, fitted by SGD
        return SGDClassifier(loss="log_loss", penalty="elasticnet", class_weight=class_weight,
                             random_state=seed, **p)
    if family == "svm":
        return SGDClassifier(loss="hinge", penalty="elasticnet", class_weight=class_weight,
                             random_state=seed, **p)
    if family == "rf":
        return RandomForestClassifier(class_weight=class_weight, n_jobs=-1,
                                      random_state=seed, **p)
    if family == "hgb":
        return HistGradientBoostingClassifier(class_weight=class_weight, early_stopping=False,
                                              random_state=seed, **p)
    if family == "ffnn":
        return KerasMLP(class_weight=class_weight, random_state=seed, **p)
    raise ValueError(f"unknown family {family!r}")


def fit(model, X, y, X_val=None, y_val=None):
    if isinstance(model, KerasMLP):
        return model.fit(X, y, X_val, y_val)
    return model.fit(X, y)


def scores(model, X) -> np.ndarray:
    """Continuous scores: probabilities where the model has them, margins for the SVM."""
    if isinstance(model, KerasMLP) or (isinstance(model, SGDClassifier) and model.loss == "hinge"):
        return np.asarray(model.decision_function(X), dtype=np.float64)
    return model.predict_proba(X)[:, 1].astype(np.float64)


def extras(model, names) -> dict:
    """Interpretable by-products, kept for Notebook 04."""
    if isinstance(model, (LogisticRegression, SGDClassifier)):
        coef = model.coef_.ravel()
        return {"coefficients": dict(zip(names, map(float, coef))),
                "n_zero_coefficients": int((coef == 0).sum()),
                "n_iter": int(np.max(model.n_iter_))}
    if isinstance(model, RandomForestClassifier):
        return {"impurity_importance": dict(zip(names, map(float, model.feature_importances_)))}
    if isinstance(model, KerasMLP):
        return {"epochs": model.epochs_, "best_val_ap_during_training": float(model.best_val_ap_)}
    return {}

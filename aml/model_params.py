"""Model training parameters (Sections 4.7-4.9, revised before any model was fitted).

Revisions of Section 4.8, agreed before the sweep: the hyperparameter search is run on a
uniform 25 % subsample of the training block (a computational measure: the class ratio is
preserved in expectation, nothing is resampled), the number of candidates per family is
reduced from sixty, and the selected configuration is fitted on the training block only
(no refit on training plus validation, so the threshold chosen on validation belongs to
the model that is tested). After the timing run (projected 20 hours, before any score was
computed): logistic regression is fitted by stochastic gradient descent like the SVM (the
saga solver needed over two minutes per search fit), the boosting grid is capped at 200
iterations, the network uses a fixed batch of 4,096 and at most 15 epochs, and the boosting
and network searches draw 10 and 8 candidates. Change nothing here once the sweep has started.
"""
from scipy.stats import loguniform

FAMILIES = ("logreg", "svm", "rf", "hgb", "ffnn")
SETS = ("A", "B", "C")
N_CANDIDATES = {"logreg": 20, "svm": 20, "rf": 10, "hgb": 10, "ffnn": 8}
SEARCH_FRACTION = 0.25        # share of training rows used for the search
SEARCH_SEED = 0               # candidate sampling and the search subsample
N_SEEDS = 5                   # final fits per selected configuration
ALERT_BUDGET = 0.001          # highest-scoring 0.1 % of transactions (Section 4.7)

# Declared search spaces (Appendix A). Every candidate uses class_weight="balanced".
GRIDS = {
    "logreg": {"alpha": loguniform(1e-7, 1e-2), "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0]},
    "svm": {"alpha": loguniform(1e-7, 1e-2), "l1_ratio": [0.0, 0.15, 0.5, 0.85, 1.0]},
    "rf": {"n_estimators": [100, 200], "max_depth": [8, 12, 16, 24],
           "min_samples_leaf": [10, 50, 200], "max_features": ["sqrt", 0.3, 0.5],
           "max_samples": [0.05, 0.1, 0.2]},
    "hgb": {"learning_rate": loguniform(0.03, 0.3), "max_iter": [100, 200],
            "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [20, 100, 500],
            "l2_regularization": loguniform(1e-3, 10)},
    "ffnn": {"hidden_layers": [1, 2], "units": [32, 64, 128], "dropout": [0.0, 0.2],
             "learning_rate": loguniform(1e-4, 3e-3)},
}
# Settings held fixed for every candidate of a family.
FIXED = {
    "logreg": {"max_iter": 30, "tol": 1e-4},
    "svm": {"max_iter": 30, "tol": 1e-4},
    "rf": {},
    "hgb": {},
    "ffnn": {"max_epochs": 15, "patience": 3, "batch_size": 4096},
}

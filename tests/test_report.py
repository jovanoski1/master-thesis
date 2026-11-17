import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from aml import report


def test_weighted_ap_matches_scikit_learn():
    rng = np.random.default_rng(0)
    y = (rng.random(3000) < 0.05).astype(int)
    s = np.round(rng.random(3000) + y * 0.3, 2)            # rounded: many ties
    w = rng.poisson(1.0, 3000).astype(float)
    p = report.SortedScores(y, s)
    assert p.ap() == pytest.approx(average_precision_score(y, s))
    assert p.ap(w) == pytest.approx(average_precision_score(y, s, sample_weight=w))


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(int)
    base = rng.random(4000)
    scores = {}
    for sd in range(3):
        noise = rng.normal(0, 0.05, 4000)
        scores[("m", "A", sd)] = base + noise
        scores[("m", "B", sd)] = base + noise + 0.8 * y          # clearly better
        scores[("m", "C", sd)] = base + noise + 0.8 * y          # identical to B
    out = report.paired_bootstrap(y, scores, [("m", "B", "A"), ("m", "C", "B")], n_boot=200)
    ba, cb = out.iloc[0], out.iloc[1]
    assert ba.difference > 0 and ba.interval_excludes_0
    assert cb.difference == pytest.approx(0) and not cb.interval_excludes_0


def test_friedman_nemenyi():
    t = pd.DataFrame({"x": [0.9, 0.8, 0.85, 0.7], "y": [0.5, 0.4, 0.45, 0.3], "z": [0.6, 0.65, 0.5, 0.35]})
    r = report.friedman_nemenyi(t)
    assert list(r["average_ranks"]) == ["x", "z", "y"] and r["average_ranks"]["x"] == 1.0
    assert r["critical_difference"] == pytest.approx(2.343 * np.sqrt(3 * 4 / 24))


def test_typology_recall():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    typ = np.array(["cycle", "cycle", "no pattern", "x", "x", "x", "x", "x"])
    s = {0: np.array([.9, .1, .8, .2, .3, .1, .1, .1])}
    out = report.typology_recall(y, typ, s, {0: 0.5}, budget=0.25)    # top 2 rows flagged
    assert out.loc["cycle", "recall_at_f1"] == 0.5 and out.loc["no pattern", "recall_in_top"] == 1.0

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from aml import evaluation as ev
from aml import models, sweep


def test_thresholds_and_metrics():
    y = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    s = np.array([.1, .2, .9, .3, .8, .4, .35, .05, .15, .25])
    tau = ev.threshold_f1(y, s)
    m = ev.at_threshold(y, s, tau)
    assert m["f1"] == pytest.approx(max(ev.at_threshold(y, s, t)["f1"] for t in s))
    tb = ev.threshold_budget(s, 0.2)                     # top 2 of 10
    assert tb == 0.8 and ev.at_threshold(y, s, tb)["alert_rate"] == 0.2
    out = ev.evaluate(y, s, tau, tb, default=0.5, budget=0.2)
    assert out["n_pos"] == 3 and out["at_default"]["tp"] == 2
    assert out["top_k"]["k"] == 2 and out["top_k"]["tp"] == 2
    tied = ev.top_k(np.array([1, 0, 0, 0]), np.array([.5, .5, .5, .5]), 0.5)
    assert tied["k"] == 2 and tied["tp"] in (0, 1)        # exactly k flagged despite ties


def test_preprocessor_uses_training_rows_only():
    names = ["amount_paid_base", "hour", "src_out_cnt", "src_in_share"]
    assert models.log_columns(names) == [0, 2]
    tr = np.array([[10, 1, 1, .5], [100, 2, 3, .2], [1000, 3, 7, .9]], dtype=np.float32)
    te = np.array([[1e6, 23, 100, .1]], dtype=np.float32)
    pre = models.Preprocessor(names).fit(tr)
    Z = pre.transform(tr)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-6)
    assert pre.transform(te)[0, 0] > 3                  # far outside training range, not rescaled to it


def toy_features(tmp_path, n=6000, seed=0, tail=True):
    rng = np.random.default_rng(seed)
    X = {"amount_paid_base": rng.lognormal(6, 2, n), "hour": rng.integers(0, 24, n).astype(float),
         "src_out_cnt": rng.poisson(3, n).astype(float), "src_in_deg": rng.poisson(2, n).astype(float)}
    logit = -4 + 0.8 * np.log1p(X["src_out_cnt"]) + 0.5 * np.log1p(X["src_in_deg"]) \
        + 0.2 * (np.log1p(X["amount_paid_base"]) - 6)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    block = np.where(np.arange(n) < .7 * n, "train", np.where(np.arange(n) < .85 * n, "val", "test"))
    df = pd.DataFrame({"file_row": np.arange(n), "block": pd.Categorical(block, ["train", "val", "test"]),
                       "is_laundering": y, "burn_in": np.arange(n) < .05 * n,
                       "tail": (np.arange(n) >= .99 * n) & tail, "snapshot_minute": 0,
                       **{k: v.astype(np.float32) for k, v in X.items()}})
    (tmp_path / "features").mkdir()
    df.to_parquet(tmp_path / "features" / "HI-Small.parquet", index=False)
    sets = {"A": ["amount_paid_base", "hour"]}
    sets["B"] = sets["A"] + ["src_out_cnt"]
    sets["C"] = sets["B"] + ["src_in_deg"]
    (tmp_path / "feature_sets_HI-Small.json").write_text(json.dumps({"sets": sets}))
    return sweep.load_variant("HI-Small", tmp_path / "features", tmp_path, 0.5, 0)


def test_sweep_resumes_and_summarises(tmp_path):
    data = toy_features(tmp_path)
    assert len(data.y_tail) == 60 and data.X_tr.shape[1] == 4 and data.sets["A"] == [0, 1]
    store = sweep.Store(tmp_path / "models")
    log = []
    for fam in ["logreg", "svm", "rf", "hgb"]:
        sweep.run_family(store, {"HI-Small": data}, fam, sets=("A", "C"),
                         n_candidates=2, n_seeds=2, progress=log.append)
    n_search, n_final = len(store.records["search"]), len(store.records["final"])
    assert (n_search, n_final) == (4 * 2 * 2, 4 * 2 * 2)
    # a second run (as after a crash) fits nothing new
    store2 = sweep.Store(tmp_path / "models")
    sweep.run_family(store2, {"HI-Small": data}, "rf", sets=("A", "C"), n_candidates=2,
                     n_seeds=2, progress=log.append)
    assert len(store2.records["final"]) == n_final
    summ = sweep.summary(store2)
    assert summ.shape[0] == 8 and ("test AP", "mean") in summ.columns
    rec = store2.records["final"]["HI-Small|logreg|C|seed0"]
    assert rec["tail"]["n"] == 60 and "coefficients" in rec["extras"]
    assert (tmp_path / "models" / "scores" / "HI-Small_rf_C_seed1.npz").exists()
    sup = sweep.run_supplement(store2, data, n_seeds=2, progress=log.append)
    assert len(sup) == 3


@pytest.mark.skipif(importlib.util.find_spec("keras") is None, reason="keras not installed")
def test_keras_network_trains_with_early_stopping(tmp_path, monkeypatch):
    data = toy_features(tmp_path)
    store = sweep.Store(tmp_path / "models")
    import aml.model_params as MP
    monkeypatch.setitem(MP.FIXED, "ffnn", {"max_epochs": 3, "patience": 1})
    sweep.run_family(store, {"HI-Small": data}, "ffnn", sets=("C",), n_candidates=1,
                     n_seeds=1, progress=lambda m: None)
    rec = store.records["final"]["HI-Small|ffnn|C|seed0"]
    assert 1 <= rec["extras"]["epochs"] <= 3 and rec["test"]["ap"] is not None


def test_sweep_without_tail_rows(tmp_path):
    data = toy_features(tmp_path, tail=False)
    store = sweep.Store(tmp_path / "models")
    sweep.run_family(store, {"HI-Small": data}, "logreg", sets=("A",), n_candidates=1,
                     n_seeds=1, progress=lambda m: None)
    assert store.records["final"]["HI-Small|logreg|A|seed0"]["tail"] is None

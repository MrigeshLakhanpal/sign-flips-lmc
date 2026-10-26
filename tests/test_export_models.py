"""Made-up data test for src/export_models.py (no real models used)."""
import hashlib
import json
import pathlib
import pickle
import sys
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import export_models as em


def _fake(tmp_path):
    rng = np.random.default_rng(0)
    runs = []
    for h in (3, 4):
        for s in (1, 0):  # deliberately out of order
            runs.append(dict(width=h, seed=s, params=dict(W1=rng.normal(size=(h, 2)), b1=rng.normal(size=h),
                                                       w2=rng.normal(size=h), b2=np.array([rng.normal()])),
                             train_loss=0.1, train_err=0.01 * s, test_loss=0.2, test_err=0.02, train_err_ckpt=0.01))
    D = dict(task="T3", mu=np.zeros(2), sd=np.ones(2), vertices=rng.normal(size=(3, 2)), runs=runs,
             data=dict(Xtr=rng.normal(size=(5, 2)), ytr=np.ones(5), Xte=rng.normal(size=(5, 2)), yte=np.zeros(5)))
    f = tmp_path / "models_T3.pkl"
    f.write_bytes(pickle.dumps(D))
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    (tmp_path / "stage1_frozen.json").write_text(json.dumps(dict(tasks=dict(T3=dict(file=f.name, sha256=sha)))))
    return D


def test_export_roundtrip(tmp_path):
    D = _fake(tmp_path)
    em.main(tmp_path)
    A = np.load(tmp_path / "models_npz" / "models_T3.npz", allow_pickle=False)
    assert A["w3_W1"].shape == (2, 3, 2) and A["w4_w2"].shape == (2, 4)
    r = next(r for r in D["runs"] if r["width"] == 4 and r["seed"] == 0)
    assert list(A["w4_seed"]) == [0, 1]  # sorted by seed
    np.testing.assert_array_equal(A["w4_W1"][0], r["params"]["W1"])
    assert A["w4_b2"][0] == r["params"]["b2"][0]


def test_refuses_changed_file(tmp_path):
    _fake(tmp_path)
    with open(tmp_path / "models_T3.pkl", "ab") as f:
        f.write(b"x")
    with pytest.raises(SystemExit):
        em.main(tmp_path)

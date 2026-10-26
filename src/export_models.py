"""Export the frozen Study 2 model files to plain NumPy archives (v2.1 release).

The analysis reads results_v2/models_<task>.pkl, whose SHA-256 fingerprints were
committed in results_v2/stage1_frozen.json before any barrier was computed. Loading a
pickle file can execute code, so for readers who only want the weights this script
writes the same networks to results_v2/models_npz/models_<task>.npz, which contain
arrays only and load with np.load(..., allow_pickle=False).

It first checks every .pkl against its committed fingerprint and stops on a mismatch.

Layout of each .npz, for every trained width h:
  w{h}_W1 (n, h, 2)  input weights      w{h}_b1 (n, h)  hidden biases
  w{h}_w2 (n, h)     output weights     w{h}_b2 (n,)    output bias
  w{h}_seed, w{h}_train_err, w{h}_test_err, w{h}_train_loss, w{h}_test_loss,
  w{h}_train_err_ckpt (n,)
plus mu, sd (standardization), vertices (task polygon) and Xtr, ytr, Xte, yte.

Usage:  python src/export_models.py results_v2
Written by Claude (26 Oct 2026); tested on made-up data in tests/test_export_models.py.
"""
import hashlib
import json
import pathlib
import pickle
import sys
import numpy as np

SCALARS = ["seed", "train_err", "test_err", "train_loss", "test_loss", "train_err_ckpt"]


def check_fingerprints(d):
    frozen = json.load(open(d / "stage1_frozen.json"))["tasks"]
    for task, v in frozen.items():
        f = d / v["file"]
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != v["sha256"]:
            raise SystemExit(f"{f}: SHA-256 {got} does not match the frozen {v['sha256']}; stopping.")
        print(f"{f.name}: fingerprint matches stage1_frozen.json")
    return frozen


def to_arrays(D):
    out = {k: np.asarray(D[k]) for k in ("mu", "sd", "vertices") if D.get(k) is not None}
    out.update({k: np.asarray(v) for k, v in D["data"].items()})
    for h in sorted({r["width"] for r in D["runs"]}):
        R = sorted((r for r in D["runs"] if r["width"] == h), key=lambda r: r["seed"])
        out[f"w{h}_W1"] = np.stack([np.asarray(r["params"]["W1"]).reshape(h, -1) for r in R])
        out[f"w{h}_b1"] = np.stack([np.asarray(r["params"]["b1"]).reshape(h) for r in R])
        out[f"w{h}_w2"] = np.stack([np.asarray(r["params"]["w2"]).reshape(h) for r in R])
        out[f"w{h}_b2"] = np.array([float(np.asarray(r["params"]["b2"]).reshape(-1)[0]) for r in R])
        for k in SCALARS:
            out[f"w{h}_{k}"] = np.array([r[k] for r in R])
    return out


def main(root="results_v2"):
    d = pathlib.Path(root)
    frozen = check_fingerprints(d)
    outdir = d / "models_npz"
    outdir.mkdir(exist_ok=True)
    for task, v in frozen.items():
        D = pickle.load(open(d / v["file"], "rb"))
        A = to_arrays(D)
        np.savez_compressed(outdir / f"models_{task}.npz", **A)
        print(f"wrote {outdir / f'models_{task}.npz'} ({len(D['runs'])} networks)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_v2")

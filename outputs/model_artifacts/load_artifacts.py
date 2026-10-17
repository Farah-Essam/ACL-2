# Written by f1_podium_prediction.ipynb, section 6.12. Usage:
#   from load_artifacts import load_model_artifacts
#   models, manifest = load_model_artifacts('outputs/model_artifacts')
#   p = models['A_logistic_regression']['predict'](rows)   # rows: DataFrame with the 18 feature columns

import hashlib, json
from pathlib import Path

import cloudpickle
import numpy as np
import torch


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build_network(architecture):
    layers, size = [], architecture["n_inputs"]
    for h in architecture["hidden"]:
        layers += [torch.nn.Linear(size, h), torch.nn.ReLU(), torch.nn.Dropout(architecture["dropout"])]
        size = h
    layers.append(torch.nn.Linear(size, 1))
    return torch.nn.Sequential(*layers)


# Load every saved frozen model from `folder` (outputs/model_artifacts).
# Returns ({name: {"key", "predict", "threshold"}}, manifest). predict(rows) takes a DataFrame with the 18 feature
# columns (missing values allowed) and returns podium probabilities. Every file is checked against its SHA-256.
def load_model_artifacts(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    columns = manifest["input columns"]
    loaded = {}
    for name, m in manifest["models"].items():
        for role, f in m["files"].items():
            assert _sha256(folder / f["file"]) == f["sha256"], f"{name}: {f['file']} does not match the manifest"
        load = lambda role: cloudpickle.loads((folder / m["files"][role]["file"]).read_bytes())
        if m["format"] == "pytorch":
            pre = load("preprocessor")
            net = _build_network(m["architecture"])
            net.load_state_dict(torch.load(folder / m["files"]["weights"]["file"], weights_only=True))
            net.eval()

            def predict(rows, pre=pre, net=net):
                X = torch.tensor(np.asarray(pre.transform(rows[columns]), dtype=np.float32))
                with torch.no_grad():
                    return torch.sigmoid(net(X).squeeze(1)).numpy().astype(float)
        elif m["format"] == "sklearn pipeline":
            pipe = load("pipeline")

            def predict(rows, pipe=pipe):
                return pipe.predict_proba(rows[columns])[:, 1]
        else:
            obj = load("preprocessor and model")

            def predict(rows, obj=obj):
                return obj["model"].predict_proba(obj["preprocessor"].transform(rows[columns]))[:, 1]
        loaded[name] = {"key": tuple(m["key"]), "predict": predict, "threshold": float.fromhex(m["threshold (exact hex)"])}
    return loaded, manifest

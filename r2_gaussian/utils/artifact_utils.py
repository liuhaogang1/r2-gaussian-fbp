"""Content identities and atomic writes for project-added artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, default=str, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def npy_path(path):
    path = Path(path)
    return path if str(path).endswith(".npy") else Path(str(path) + ".npy")


def atomic_npy(path, value):
    path = npy_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **values)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def dataset_identity(dataset):
    """Bind training projections to geometry, independent of record order."""
    dataset = Path(dataset)
    if dataset.is_file():
        return identity(file_identity(dataset))
    data = json.loads((dataset / "meta_data.json").read_text(encoding="utf-8"))
    records = []
    for record in data["proj_train"]:
        name = str(record["file_path"]).replace("\\", "/")
        records.append({"name": name, "angle": float(record["angle"]),
                        "content": file_identity(dataset / name)["sha256"]})
    return identity({"scanner": data["scanner"],
                     "projections": sorted(records, key=lambda item: item["name"])})

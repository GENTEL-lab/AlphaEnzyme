import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExternalFeatureBank:
    path: str
    feature_keys: tuple[str, ...]
    id_to_index: dict[str, int]
    feature_dim: int
    data: dict[str, np.ndarray] | None = field(default=None, repr=False)

    def _ensure_data(self):
        # NOTE:
        # NpzFile.__getitem__ will reopen and inflate the requested array from the
        # zip archive on every access. For per-sample training reads this becomes a
        # severe CPU bottleneck. We therefore materialize the feature arrays once in
        # the parent process and let DataLoader workers share them through fork/COW.
        if self.data is None:
            with np.load(self.path, allow_pickle=True) as data:
                self.data = {
                    key: np.asarray(data[key]).astype(np.float16, copy=False)
                    for key in self.feature_keys
                }
        return self.data

    def contains(self, sample_id: str) -> bool:
        return str(sample_id) in self.id_to_index

    def get(self, sample_id: str) -> np.ndarray | None:
        idx = self.id_to_index.get(str(sample_id))
        if idx is None:
            return None
        data = self._ensure_data()
        parts = [data[key][idx] for key in self.feature_keys]
        if len(parts) == 1:
            return np.asarray(parts[0], dtype=np.float16)
        return np.concatenate(parts, axis=-1).astype(np.float16, copy=False)


def _resolve_primary_feature(data, bank_path: str) -> np.ndarray:
    if "s_mean" in data:
        return data["s_mean"]
    if "enzyme_s_pool" in data:
        return data["enzyme_s_pool"]
    raise KeyError(f"{bank_path} missing s_mean/enzyme_s_pool")


def _resolve_feature_keys(data, bank_path: str, feature_keys=None, include_z: bool = False) -> list[str]:
    if feature_keys:
        resolved = []
        for key in feature_keys:
            if key not in data:
                raise KeyError(f"{bank_path} missing requested feature key: {key}")
            resolved.append(str(key))
        return resolved

    keys = []
    if "s_mean" in data:
        keys.append("s_mean")
    elif "enzyme_s_pool" in data:
        keys.append("enzyme_s_pool")
    else:
        raise KeyError(f"{bank_path} missing s_mean/enzyme_s_pool")
    if include_z:
        if "z_mean" not in data:
            raise KeyError(f"{bank_path} missing z_mean while include_z=True")
        keys.append("z_mean")
    return keys


def load_external_feature_bank(
    bank_path: str | None,
    include_z: bool = False,
    feature_keys=None,
    preload: bool = True,
) -> ExternalFeatureBank | None:
    if not bank_path:
        return None
    if not os.path.exists(bank_path):
        raise FileNotFoundError(f"External feature bank not found: {bank_path}")

    data = np.load(bank_path, allow_pickle=True, mmap_mode="r")
    sample_ids = data["sample_ids"].astype(str)
    resolved_keys = _resolve_feature_keys(data, bank_path, feature_keys=feature_keys, include_z=include_z)
    id_to_index = {sample_id: idx for idx, sample_id in enumerate(sample_ids.tolist())}
    feature_dim = 0
    for key in resolved_keys:
        arr = data[key]
        if arr.ndim == 1:
            feature_dim += 1
        else:
            feature_dim += int(arr.shape[-1])
    data.close()
    bank = ExternalFeatureBank(
        path=bank_path,
        feature_keys=tuple(str(key) for key in resolved_keys),
        id_to_index=id_to_index,
        feature_dim=int(feature_dim),
    )
    if preload:
        bank._ensure_data()
    return bank


def fetch_external_feature(
    bank: ExternalFeatureBank | None,
    sample_id: str,
    missing_policy: str = "error",
    dtype=np.float32,
) -> np.ndarray:
    if bank is None:
        return np.zeros((0,), dtype=dtype)

    feature = bank.get(sample_id)
    if feature is None:
        if missing_policy == "zero":
            return np.zeros((bank.feature_dim,), dtype=dtype)
        raise KeyError(f"sample_id={sample_id} missing in external bank: {bank.path}")

    return feature.astype(dtype, copy=False)


def normalize_bank_specs(raw_specs) -> list[dict]:
    if raw_specs is None:
        return []
    if isinstance(raw_specs, (str, os.PathLike)):
        raw_specs = [str(raw_specs)]

    normalized = []
    for idx, raw_spec in enumerate(raw_specs):
        if raw_spec is None:
            continue
        if isinstance(raw_spec, (str, os.PathLike)):
            path = str(raw_spec)
            normalized.append(
                {
                    "name": os.path.splitext(os.path.basename(path))[0] or f"bank_{idx}",
                    "path": path,
                    "include_z": False,
                    "keys": None,
                }
            )
            continue
        if not isinstance(raw_spec, dict):
            raise TypeError(f"Unsupported external feature bank spec: {type(raw_spec)!r}")

        spec = dict(raw_spec)
        path = spec.get("path") or spec.get("bank_path")
        if not path:
            raise ValueError(f"External feature bank spec missing path: {raw_spec}")
        normalized.append(
            {
                "name": str(spec.get("name") or os.path.splitext(os.path.basename(path))[0] or f"bank_{idx}"),
                "path": str(path),
                "include_z": bool(spec.get("include_z", False)),
                "keys": list(spec.get("keys")) if spec.get("keys") else None,
            }
        )
    return normalized


def load_aligned_external_feature_matrix(
    sample_ids,
    bank_specs=None,
    missing_policy: str = "error",
):
    specs = normalize_bank_specs(bank_specs)
    if not specs:
        return None, []

    ordered_sample_ids = [str(sample_id) for sample_id in sample_ids]
    aligned_features = []
    feature_meta = []

    for spec in specs:
        bank = load_external_feature_bank(
            spec["path"],
            include_z=bool(spec.get("include_z", False)),
            feature_keys=spec.get("keys"),
        )
        matrix = np.zeros((len(ordered_sample_ids), bank.feature_dim), dtype=np.float16)
        missing = []
        for idx, sample_id in enumerate(ordered_sample_ids):
            feature = bank.get(sample_id)
            if feature is None:
                if missing_policy == "error":
                    missing.append(sample_id)
                continue
            matrix[idx] = feature
        if missing:
            preview = missing[:10]
            raise KeyError(
                f"Missing {len(missing)} sample_ids in external bank {spec['path']}, "
                f"examples: {preview}"
            )
        aligned_features.append(matrix)
        feature_meta.append(
            {
                "name": spec["name"],
                "path": spec["path"],
                "include_z": bool(spec.get("include_z", False)),
                "keys": spec.get("keys"),
                "dim": int(bank.feature_dim),
            }
        )

    combined = np.concatenate(aligned_features, axis=-1).astype(np.float16, copy=False)
    return combined, feature_meta

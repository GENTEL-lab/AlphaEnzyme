import numpy as np
import torch


def _load_npz(path: str):
    return np.load(path, allow_pickle=True)


class ExternalFeatureStore:
    def __init__(self, bank_path: str, include_z: bool = False, missing_policy: str = "zero"):
        data = _load_npz(bank_path)
        sample_ids = data["sample_ids"].astype(str)
        features = [data["s_mean"].astype(np.float16, copy=False)]
        if include_z:
            if "z_mean" not in data:
                raise KeyError(f"include_z=True but z_mean not found in {bank_path}")
            features.append(data["z_mean"].astype(np.float16, copy=False))
        self.sample_ids = sample_ids
        self.features = np.concatenate(features, axis=-1).astype(np.float16, copy=False)
        self.id_to_idx = {sid: idx for idx, sid in enumerate(self.sample_ids.tolist())}
        self.feature_dim = int(self.features.shape[1])
        self.missing_policy = str(missing_policy).lower()

    def get(self, sample_id: str) -> torch.Tensor:
        idx = self.id_to_idx.get(str(sample_id))
        if idx is None:
            if self.missing_policy == "zero":
                return torch.zeros(self.feature_dim, dtype=torch.float32)
            raise KeyError(f"sample_id={sample_id} not found in external feature bank")
        return torch.from_numpy(self.features[idx].astype(np.float32, copy=False)).float()


class DirectContextAssetStore:
    def __init__(self, cache_path: str, bank_path: str, missing_policy: str = "error"):
        cache = _load_npz(cache_path)
        bank = _load_npz(bank_path)

        self.query_ids = cache["sample_ids"].astype(str)
        self.query_to_idx = {sid: idx for idx, sid in enumerate(self.query_ids.tolist())}
        self.context_indices = cache["context_indices"].astype(np.int32, copy=False)
        self.context_mask = cache["context_mask"].astype(bool, copy=False)
        self.context_scores = cache["pair_corr_scores"].astype(np.float32, copy=False)
        self.overall_corr_max = cache["overall_corr_max"] if "overall_corr_max" in cache.files else None
        self.overall_corr_mean = cache["overall_corr_mean"] if "overall_corr_mean" in cache.files else None
        if self.overall_corr_max is None:
            self.overall_corr_max = np.zeros((len(self.query_ids),), dtype=np.float32)
        else:
            self.overall_corr_max = self.overall_corr_max.astype(np.float32, copy=False)
        if self.overall_corr_mean is None:
            self.overall_corr_mean = np.zeros((len(self.query_ids),), dtype=np.float32)
        else:
            self.overall_corr_mean = self.overall_corr_mean.astype(np.float32, copy=False)

        self.bank_sample_ids = bank["sample_ids"].astype(str)
        if "context_features" in bank:
            bank_features = bank["context_features"]
        elif "pair_features" in bank:
            bank_features = bank["pair_features"]
        else:
            raise KeyError(f"{bank_path} missing context_features/pair_features")
        self.bank_features = bank_features.astype(np.float16, copy=False)
        self.context_len = int(self.context_indices.shape[1])
        self.context_dim = int(self.bank_features.shape[1])
        self.missing_policy = str(missing_policy).lower()

    def get(self, sample_id: str):
        query_idx = self.query_to_idx.get(str(sample_id))
        if query_idx is None:
            if self.missing_policy == "zero":
                return (
                    torch.zeros((self.context_len, self.context_dim), dtype=torch.float32),
                    torch.zeros((self.context_len,), dtype=torch.float32),
                    torch.zeros((self.context_len,), dtype=torch.float32),
                    torch.zeros((2,), dtype=torch.float32),
                )
            raise KeyError(f"sample_id={sample_id} not found in direct context cache")

        indices = self.context_indices[query_idx]
        mask = self.context_mask[query_idx]
        scores = self.context_scores[query_idx]
        features = np.zeros((self.context_len, self.context_dim), dtype=np.float32)
        valid = mask & (indices >= 0)
        if np.any(valid):
            features[valid] = self.bank_features[indices[valid]].astype(np.float32, copy=False)

        overall = np.asarray(
            [
                float(self.overall_corr_max[query_idx]),
                float(self.overall_corr_mean[query_idx]),
            ],
            dtype=np.float32,
        )
        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(mask.astype(np.float32, copy=False)).float(),
            torch.from_numpy(scores.astype(np.float32, copy=False)).float(),
            torch.from_numpy(overall).float(),
        )

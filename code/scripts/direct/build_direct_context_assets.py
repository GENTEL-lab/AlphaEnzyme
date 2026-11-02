from collections import defaultdict
import numpy as np
from scripts.rerank.retrieval_context_common import align_external_similarity_features,build_reaction_neighbor_cache,get_or_encode_rxn_fp,l2_normalize,load_external_similarity_bank,load_rxn2fp

def build_query_cache(
    query_table,
    query_external_bank_path,
    include_query_z,
    positive_bank,
    reaction_topk,
    reaction_preselect,
    top_enzymes_per_reaction,
    max_context,
    exclude_self_sample,
    rxn2fp,
    num_workers,
    similarity_mode,
):
    query_external_bank = load_external_similarity_bank(query_external_bank_path)
    query_enzyme_features = align_external_similarity_features(
        sample_ids=query_table["sample_ids"],
        external_bank=query_external_bank,
        include_z=bool(include_query_z),
    ).astype(np.float32, copy=False)
    query_enzyme_features = l2_normalize(query_enzyme_features)

    train_positive_reactions = sorted(set(positive_bank["reactions"].tolist()))
    query_reactions = sorted(set(query_table["reactions"].tolist()))
    reaction_neighbor_cache = build_reaction_neighbor_cache(
        query_reactions=query_reactions,
        train_reactions=train_positive_reactions,
        rxn2fp=rxn2fp,
        topk=int(reaction_topk),
        preselect=int(reaction_preselect),
        exclude_self=False,
        num_workers=int(num_workers),
        similarity_mode=str(similarity_mode),
    )

    n = len(query_table["sample_ids"])
    context_indices = np.full((n, max_context), -1, dtype=np.int32)
    context_mask = np.zeros((n, max_context), dtype=bool)
    pair_corr_scores = np.zeros((n, max_context), dtype=np.float32)

    grouped_query = defaultdict(list)
    for idx, reaction in enumerate(query_table["reactions"].tolist()):
        grouped_query[str(reaction)].append(idx)

    pos_ids = positive_bank["sample_ids"].astype(str)
    pos_enzyme_features = positive_bank["enzyme_features"].astype(np.float32, copy=False)
    reaction_to_local_pos = positive_bank["reaction_to_local_pos"]

    for query_reaction, query_indices_list in grouped_query.items():
        q_indices = np.asarray(query_indices_list, dtype=np.int64)
        query_ids = query_table["sample_ids"][q_indices].astype(str)
        query_enzyme_mat = query_enzyme_features[q_indices]

        slot = 0
        for neighbor_rxn, rxn_score in reaction_neighbor_cache.get(query_reaction, []):
            local_pos = reaction_to_local_pos.get(str(neighbor_rxn))
            if local_pos is None or local_pos.size == 0:
                continue

            sims = query_enzyme_mat @ pos_enzyme_features[local_pos].T
            if exclude_self_sample:
                candidate_ids = pos_ids[local_pos]
                same_mask = query_ids[:, None] == candidate_ids[None, :]
                sims = np.where(same_mask, -1e9, sims)

            valid_per_row = (sims > -1e8).sum(axis=1)
            max_valid = int(valid_per_row.max()) if valid_per_row.size else 0
            topn = min(
                int(top_enzymes_per_reaction),
                int(local_pos.shape[0]),
                int(max_context - slot),
                int(max_valid),
            )
            if topn <= 0:
                continue

            part = np.argpartition(sims, -topn, axis=1)[:, -topn:]
            part_scores = np.take_along_axis(sims, part, axis=1)
            order = np.argsort(part_scores, axis=1)[:, ::-1]
            top_pos = np.take_along_axis(part, order, axis=1)[:, :topn]
            top_scores = np.take_along_axis(part_scores, order, axis=1)[:, :topn]
            picked_local = local_pos[top_pos]
            picked_pair_scores = float(rxn_score) * top_scores.astype(np.float32, copy=False)

            take = min(topn, int(max_context - slot))
            cols = np.arange(slot, slot + take)[None, :]
            context_indices[q_indices[:, None], cols] = picked_local[:, :take].astype(np.int32, copy=False)
            valid_top = top_scores[:, :take] > -1e8
            context_mask[q_indices[:, None], cols] = valid_top
            pair_corr_scores[q_indices[:, None], cols] = np.where(
                valid_top,
                picked_pair_scores[:, :take],
                0.0,
            )
            slot += take
            if slot >= max_context:
                break

    overall_corr_max = np.where(context_mask.any(axis=1), pair_corr_scores.max(axis=1), 0.0).astype(np.float32, copy=False)
    overall_corr_mean = np.divide(
        pair_corr_scores.sum(axis=1),
        context_mask.sum(axis=1).clip(min=1),
        dtype=np.float32,
    ).astype(np.float32, copy=False)

    return {
        "sample_ids": query_table["sample_ids"].astype(str),
        "reactions": query_table["reactions"].astype(str),
        "labels": query_table["labels"].astype(np.int64),
        "context_indices": context_indices,
        "context_mask": context_mask,
        "pair_corr_scores": pair_corr_scores,
        "overall_corr_max": overall_corr_max,
        "overall_corr_mean": overall_corr_mean,
    }, reaction_neighbor_cache

def load_positive_bank(bank_path):
    bank = np.load(bank_path, allow_pickle=True)
    sample_ids = bank["sample_ids"].astype(str)
    reactions = bank["reactions"].astype(str)
    if "enzyme_features" not in bank:
        raise KeyError(f"{bank_path} missing enzyme_features")
    enzyme_features = bank["enzyme_features"].astype(np.float32, copy=False)
    context_features = bank["context_features"].astype(np.float16, copy=False)
    reaction_to_local_pos = defaultdict(list)
    for idx, reaction in enumerate(reactions.tolist()):
        reaction_to_local_pos[str(reaction)].append(idx)
    return {
        "sample_ids": sample_ids,
        "reactions": reactions,
        "labels": bank["labels"].astype(np.int64, copy=False) if "labels" in bank else np.ones((sample_ids.shape[0],), dtype=np.int64),
        "enzyme_features": enzyme_features,
        "context_features": context_features,
        "reaction_to_local_pos": {
            reaction: np.asarray(indices, dtype=np.int64)
            for reaction, indices in reaction_to_local_pos.items()
        },
    }

"""Frozen-result metric definitions extracted unchanged from the final evaluator."""
from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def compute_dcg(rels, k=None):
    if k is None:
        k = len(rels)
    else:
        k = min(k, len(rels))
    dcg = 0
    for i in range(k):
        dcg += (2**rels[i] - 1) / np.log2(i + 2)
    return dcg

def compute_ndcg(rels, k=None):
    rels = np.asarray(rels, dtype=np.float32)
    dcg = compute_dcg(rels, k=k)
    ideal_rels = np.sort(rels)[::-1]
    idcg = compute_dcg(ideal_rels, k=k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)

def calculate_enrichment_factor(labels_sorted, top_percent):
    n = len(labels_sorted)
    topk = max(int(top_percent * n), 5)
    num_active_top = labels_sorted[:topk].sum()
    total_active = labels_sorted.sum()
    if total_active == 0:
        return 0.0
    return num_active_top / (total_active * top_percent)

def deduplicate_by_reaction_enzyme(scores, labels, sample_ids, sample_metadata, enabled=True):
    """按 (reaction, enzyme) 去重，保留 CSV 原始顺序中的第一条。"""
    raw_count = len(scores)
    if not enabled:
        return scores, labels, sample_ids, sample_metadata, {
            'Duplicate Enzyme Groups Removed': 0,
            'Duplicate Rows Removed': 0,
        }

    ordered_records = []
    for idx, meta in enumerate(sample_metadata):
        row_order = meta.get('row_order')
        ordered_records.append((
            row_order is None,
            row_order if row_order is not None else idx,
            idx,
        ))
    ordered_records.sort()

    seen_keys = set()
    duplicate_groups = set()
    kept_indices = []
    for _, _, idx in ordered_records:
        meta = sample_metadata[idx]
        reaction = meta.get('reaction')
        enzyme = meta.get('enzyme')
        if reaction is None or enzyme is None:
            kept_indices.append(idx)
            continue

        dedup_key = (reaction, enzyme)
        if dedup_key in seen_keys:
            duplicate_groups.add(dedup_key)
            continue

        seen_keys.add(dedup_key)
        kept_indices.append(idx)

    kept_indices.sort(key=lambda idx: (
        sample_metadata[idx].get('row_order') is None,
        sample_metadata[idx].get('row_order') if sample_metadata[idx].get('row_order') is not None else idx,
        idx,
    ))

    dedup_scores = scores[kept_indices]
    dedup_labels = labels[kept_indices]
    dedup_sample_ids = sample_ids[kept_indices]
    dedup_metadata = [sample_metadata[idx] for idx in kept_indices]

    stats = {
        'Duplicate Enzyme Groups Removed': int(len(duplicate_groups)),
        'Duplicate Rows Removed': int(raw_count - len(dedup_scores)),
    }
    return dedup_scores, dedup_labels, dedup_sample_ids, dedup_metadata, stats

def evaluate_ranking_by_reaction(scores, labels, sample_ids, sample_metadata, k_values=[1, 3, 5, 10, 20, 50]):
    """按反应粒度评估排名性能"""
    reaction_groups = defaultdict(list)
    n_unmapped = 0
    for i, sid in enumerate(sample_ids):
        rxn_smiles = sample_metadata[i].get('reaction')
        if rxn_smiles is not None:
            reaction_groups[rxn_smiles].append((scores[i], labels[i], sid))
        else:
            n_unmapped += 1
    if n_unmapped > 0:
        print(f"Warning: {n_unmapped} samples could not be mapped to a reaction")

    n_reactions = len(reaction_groups)
    best_positive_ranks = []
    per_reaction_recall_at_k = {k: [] for k in k_values}
    reactions_with_positive = 0
    rankable_reactions_for_auc = 0
    per_reaction_auc = []
    per_reaction_ap = []
    results = {}

    if n_reactions == 0:
        for k in k_values:
            results[f'Top-{k} Success Rate (%)'] = 0.0
            results[f'Top-{k} Recall (%)'] = 0.0
        results['MRR'] = 0.0
        results['Mean Rank'] = 0.0
        results['Median Rank'] = 0.0
        results['DCG@10'] = 0.0
        results['NDCG@10'] = 0.0
        results['EF@1%'] = 0.0
        results['EF@2%'] = 0.0
        results['_n_total_reactions'] = 0
        results['_n_reactions_with_positive'] = 0
        results['_n_rankable_reactions_for_auc'] = 0
        print("\nWarning: No reaction groups available for reaction-level evaluation")
        return results

    for rxn_smiles, items in reaction_groups.items():
        items_sorted = sorted(items, key=lambda x: -x[0])
        item_scores = np.array([score for score, _, _ in items_sorted], dtype=np.float32)
        item_labels = np.array([label for _, label, _ in items_sorted], dtype=np.int32)
        pos_ranks = []
        for rank_idx, (score, label, sid) in enumerate(items_sorted):
            if label == 1:
                pos_ranks.append(rank_idx + 1)
        if pos_ranks:
            n_pos = len(pos_ranks)
            best_rank = min(pos_ranks)
            best_positive_ranks.append(best_rank)
            reactions_with_positive += 1
            for k in k_values:
                hits = sum(1 for r in pos_ranks if r <= k)
                per_reaction_recall_at_k[k].append(hits / n_pos)

        if 0 < item_labels.sum() < len(item_labels):
            rankable_reactions_for_auc += 1
            per_reaction_auc.append(roc_auc_score(item_labels, item_scores))
            per_reaction_ap.append(average_precision_score(item_labels, item_scores))

    best_positive_ranks = np.array(best_positive_ranks)

    print("\n" + "=" * 60)
    print("Reaction-Level Ranking Evaluation")
    print("=" * 60)
    print(f"Total reactions: {n_reactions}")
    print(f"Reactions with positive samples: {reactions_with_positive}")
    print(f"Rankable reactions for AUC/AP: {rankable_reactions_for_auc}")

    group_sizes = [len(v) for v in reaction_groups.values()]
    print(f"Candidates per reaction: mean={np.mean(group_sizes):.1f}, "
          f"median={np.median(group_sizes):.1f}, "
          f"min={np.min(group_sizes)}, max={np.max(group_sizes)}")

    print(f"\n--- Top-k Success Rate (reaction-level) ---")
    for k in k_values:
        success_rate = (np.mean(best_positive_ranks <= k) * 100) if len(best_positive_ranks) > 0 else 0.0
        results[f'Top-{k} Success Rate (%)'] = success_rate
        hits = int(np.sum(best_positive_ranks <= k)) if len(best_positive_ranks) > 0 else 0
        print(f"Top-{k} Success Rate: {success_rate:.2f}% ({hits}/{len(best_positive_ranks)})")

    print(f"\n--- Top-k Recall (reaction-level) ---")
    for k in k_values:
        recall = (np.mean(per_reaction_recall_at_k[k]) * 100) if per_reaction_recall_at_k[k] else 0.0
        results[f'Top-{k} Recall (%)'] = recall
        print(f"Top-{k} Recall: {recall:.2f}%")

    mrr = float(np.mean(1.0 / best_positive_ranks)) if len(best_positive_ranks) > 0 else 0.0
    results['MRR'] = mrr
    print(f"\nMRR: {mrr:.4f}")

    mean_rank = float(np.mean(best_positive_ranks)) if len(best_positive_ranks) > 0 else 0.0
    median_rank = float(np.median(best_positive_ranks)) if len(best_positive_ranks) > 0 else 0.0
    results['Mean Rank'] = mean_rank
    results['Median Rank'] = median_rank
    print(f"Mean Rank: {mean_rank:.2f}")
    print(f"Median Rank: {median_rank:.2f}")

    if per_reaction_auc:
        mean_auc = float(np.mean(per_reaction_auc))
        mean_ap = float(np.mean(per_reaction_ap))
        results['Mean AUC (per-reaction)'] = mean_auc
        results['Mean AP (per-reaction)'] = mean_ap
        print(f"Mean AUC (per-reaction): {mean_auc:.4f}")
        print(f"Mean AP (per-reaction): {mean_ap:.4f}")

    # DCG/NDCG@10 and EF@1%/2%
    dcg_list = []
    ndcg_list = []
    ef1_list = []
    ef2_list = []
    for rxn_smiles, items in reaction_groups.items():
        items_sorted = sorted(items, key=lambda x: -x[0])
        rels = np.array([lbl for _, lbl, _ in items_sorted], dtype=np.float32)
        dcg_list.append(compute_dcg(rels, k=10))
        ndcg_list.append(compute_ndcg(rels, k=10))
        ef1_list.append(calculate_enrichment_factor(rels, top_percent=0.01))
        ef2_list.append(calculate_enrichment_factor(rels, top_percent=0.02))

    dcg10 = float(np.mean(dcg_list)) if dcg_list else 0.0
    ndcg10 = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    ef1 = float(np.mean(ef1_list)) if ef1_list else 0.0
    ef2 = float(np.mean(ef2_list)) if ef2_list else 0.0
    results['DCG@10'] = dcg10
    results['NDCG@10'] = ndcg10
    results['EF@1%'] = ef1
    results['EF@2%'] = ef2
    print(f"\nDCG@10: {dcg10:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")
    print(f"EF@1%: {ef1:.4f}")
    print(f"EF@2%: {ef2:.4f}")

    results['_n_total_reactions'] = n_reactions
    results['_n_reactions_with_positive'] = reactions_with_positive
    results['_n_rankable_reactions_for_auc'] = rankable_reactions_for_auc
    return results

def evaluate_global(scores, labels):
    """全局指标"""
    results = {}

    print("\n" + "=" * 60)
    print("Global Evaluation Results")
    print("=" * 60)

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0
    try:
        ap = average_precision_score(labels, scores)
    except Exception:
        ap = 0.0
    predictions = (scores > 0.5).astype(int)
    accuracy = np.mean(predictions == labels)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)

    results['AUC'] = auc
    results['Average Precision'] = ap
    results['Accuracy'] = accuracy
    results['Balanced Accuracy'] = balanced_acc
    results['F1'] = f1
    results['Precision'] = precision
    results['Recall'] = recall

    print(f"AUC: {auc:.4f}")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"Accuracy (threshold=0.5): {accuracy:.4f}")
    print(f"Balanced Accuracy (threshold=0.5): {balanced_acc:.4f}")
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return results

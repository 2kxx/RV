"""
Rebuild test_pointwise_<dim>.json with broader label distribution.

Compared to mine_dim_test_pointwise() in build_dim_seed_v2.py, this script:
1) Uses ALL (record, method) pairs as candidates -- NOT limited to one
   candidate per record. (The original "used_recs" cap was the main reason
   the original test set collapsed to the high-label tail: only 130 records
   * 1 candidate each, vs the true pool of 572 (record, method) candidates
   for chromaticity.)
2) Allocates target_count by EXACTLY targeting wider band coverage with
   priority for low-band candidates (which are scarce).
3) Excludes URLs that already appear as winner/loser in v2_prompt_pair.json
   (train/test isolation -- mirrors the original logic).
4) Output schema is byte-for-byte compatible with the existing
   test_pointwise_<dim>.json format (no grader changes needed).

Usage:
  python scripts/v_arr_mvp/rebuild_test_pointwise_chromaticity.py \
      --dim chromaticity \
      --target_count 30 \
      --seed 0 \
      --out data/v_arr_mvp/chromaticity_mvp/test_pointwise_chromaticity_v2.json
"""
import argparse
import json
import os
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Dimension sub-keys (mirrors scripts/v_arr_mvp/dimension_registry.py)
# ---------------------------------------------------------------------------
DIM_SUB_KEYS = {
    "chromaticity": [
        "Aesthetics-Chromaticity-Emotive_Power",
        "Aesthetics-Chromaticity-Harmony",
    ],
    "expression": [
        "Acting-Expression-Accuracy",
        "Acting-Expression-Continuity",
        "Acting-Expression-Expression_Diversity",
        "Acting-Expression-Facial_Tension",
    ],
    "logic": [
        "Multi-Shot_Cutting-Logic-Multi-shot_Rationality",
        "Multi-Shot_Cutting-Logic-Narrative_Continuity",
        "Multi-Shot_Cutting-Logic-Scene_Consistency",
        "Multi-Shot_Cutting-Logic-Spatial_Continuity",
    ],
    "rhythm": [
        "Multi-Shot_Cutting-Rhythm-Rhythmic_Layering",
        "Multi-Shot_Cutting-Rhythm-Shot_Duration_Rationality",
    ],
    "soundscape": [
        "Sound_Design-Soundscape-Ambient_Sound_Fidelity",
        "Sound_Design-Soundscape-Musical_Score_Alignment",
        "Sound_Design-Soundscape-Narrative_Sound_Design",
    ],
    # --- added 2026-05-25: 3 more low-PLCC dimensions for round-1 v2 sweep ---
    "pacing": [
        "Cinematography-Pacing-Movement_Rationality",
    ],
    "progression": [
        "Affectivity-Progression-Intensity_Layering",
        "Affectivity-Progression-Transition_Continuity",
    ],
    "consistency": [
        "Acting-Consistency-Attribute",
        "Acting-Consistency-Face_Identity",
    ],
}

DIM_META = {
    "chromaticity": {
        "evalverse_aspect": "Aesthetics",
        "evalverse_main": "Chromaticity",
    },
    "expression": {
        "evalverse_aspect": "Acting",
        "evalverse_main": "Expression",
    },
    "logic": {
        "evalverse_aspect": "Multi",
        "evalverse_main": "Shot_Cutting",
    },
    "rhythm": {
        "evalverse_aspect": "Multi",
        "evalverse_main": "Shot_Cutting",
    },
    "soundscape": {
        "evalverse_aspect": "Sound_Design",
        "evalverse_main": "Soundscape",
    },
    "pacing": {
        "evalverse_aspect": "Cinematography",
        "evalverse_main": "Pacing",
    },
    "progression": {
        "evalverse_aspect": "Affectivity",
        "evalverse_main": "Progression",
    },
    "consistency": {
        "evalverse_aspect": "Acting",
        "evalverse_main": "Consistency",
    },
}


def extract_avg(eval_result: dict, sub_key: str):
    """sub-keys are flat in evaluate_result; values are
    {scores_list, average_score, flag} dicts -- pull average_score."""
    v = eval_result.get(sub_key)
    if v is None:
        return None
    if isinstance(v, dict):
        a = v.get("average_score")
        return float(a) if a is not None else None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def label_score_from_mean_01(mean_01: float) -> float:
    return round(max(1.0, min(5.0, 1.0 + 4.0 * float(mean_01))), 2)


def collect_candidates(jsonl_paths, sub_keys, excluded_urls):
    """Return list of candidate dicts. Each (record, method) pair is one
    candidate; URL must be unique and not in excluded_urls."""
    candidates = []
    used_urls = set()
    stats = Counter()
    for jp in jsonl_paths:
        with open(jp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stats["records"] += 1
                ed = rec.get("evaluate_data") or []
                prompt_zh = rec.get("instruction_prompt") or ""
                task_name = rec.get("task_name")
                ref_path = rec.get("path") or rec.get("path_cos_url") or ""
                for ent in ed:
                    if not isinstance(ent, dict):
                        continue
                    er = ent.get("evaluate_result") or {}
                    if not isinstance(er, dict):
                        continue
                    sub_scores = {}
                    for k in sub_keys:
                        a = extract_avg(er, k)
                        if a is not None:
                            sub_scores[k] = a
                    if not sub_scores:
                        stats["no_subs"] += 1
                        continue
                    url = ent.get("cos_url") or ent.get("path")
                    if not url:
                        stats["no_url"] += 1
                        continue
                    if url in excluded_urls:
                        stats["excluded_train_url"] += 1
                        continue
                    if url in used_urls:
                        stats["dup_url"] += 1
                        continue
                    used_urls.add(url)
                    mean_01 = sum(sub_scores.values()) / len(sub_scores)
                    candidates.append({
                        "method": ent.get("method") or "unknown",
                        "url": url,
                        "subs": sub_scores,
                        "mean_01": mean_01,
                        "label": label_score_from_mean_01(mean_01),
                        "prompt_en": rec.get("instruction_prompt_en")
                                     or rec.get("prompt_en")
                                     or "",
                        "prompt_zh": prompt_zh,
                        "task_name": task_name,
                        "evalverse_ref_path": ref_path,
                    })
    return candidates, stats


def stratified_sample(candidates, target_count, seed):
    """Allocate target_count across 5 bands with PRIORITY ON LOW BANDS.

    Strategy: greedy fill from low to high; if a low band has fewer
    candidates than its quota, the unused quota cascades to the next-lowest
    available band (NOT to the high band) -- this keeps the result as flat
    as possible across the lower half.
    """
    bands = [
        (1.0, 1.8),  # band 0
        (1.8, 2.6),  # band 1
        (2.6, 3.4),  # band 2
        (3.4, 4.2),  # band 3
        (4.2, 5.001),  # band 4
    ]
    # Pool per band
    pools = [[] for _ in bands]
    for c in candidates:
        for bi, (lo, hi) in enumerate(bands):
            if lo <= c["label"] < hi:
                pools[bi].append(c)
                break
    rng = random.Random(seed)
    for p in pools:
        # sort by label first, then shuffle within label so we get a
        # spread of method/source diversity at each label level
        rng.shuffle(p)

    # Target distribution: aim for label_std maximization but constrained
    # by pool sizes.
    # Ideal per-band quota for target_count=30:
    #   band 0 (1.0-1.8): 6
    #   band 1 (1.8-2.6): 6
    #   band 2 (2.6-3.4): 6
    #   band 3 (3.4-4.2): 6
    #   band 4 (4.2-5.0): 6
    base_quota = [target_count // 5] * 5
    rem = target_count - sum(base_quota)
    for i in range(rem):
        base_quota[i % 5] += 1

    chosen = []
    final_per_band = [0] * 5
    pool_sizes = [len(p) for p in pools]

    # Pass 1: each band tries to fill its quota.
    leftover = 0
    for bi in range(5):
        take = min(base_quota[bi], len(pools[bi]))
        chosen.extend(pools[bi][:take])
        final_per_band[bi] = take
        # remember unused pool for pass 2
        pools[bi] = pools[bi][take:]
        leftover += base_quota[bi] - take

    # Pass 2: redistribute leftover quota -- prefer to fill from
    # bands 1-4 (mid+high), AVOIDING further filling band 4 unless
    # nothing else is left. This keeps the distribution wide.
    redistrib_order = [1, 2, 3, 0, 4]  # mid-low/mid first, then low, then high
    while leftover > 0:
        progressed = False
        for bi in redistrib_order:
            if leftover == 0:
                break
            if pools[bi]:
                chosen.append(pools[bi].pop(0))
                final_per_band[bi] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break

    rng.shuffle(chosen)
    chosen.sort(key=lambda x: x["label"])  # final stable order (low->high)
    return chosen, final_per_band, pool_sizes


def make_payload(chosen, dim, id_prefix):
    sub_keys = DIM_SUB_KEYS[dim]
    meta_const = DIM_META[dim]
    out = []
    for i, c in enumerate(sorted(chosen, key=lambda x: x["url"])):
        out.append({
            "id": f"{id_prefix}_{i:03d}",
            "query": c["prompt_en"] or c["prompt_zh"],
            "response": c["url"],
            "label_score": c["label"],
            "dimension": dim,
            "task_type": c["task_name"] or "t2v",
            "metadata": {
                "dimension": dim,
                "evalverse_aspect": meta_const["evalverse_aspect"],
                "evalverse_main": meta_const["evalverse_main"],
                "evalverse_sub_keys": list(sub_keys),
                "synthesis_mode": "test_pointwise_dim",
                "variant": "test_pointwise_v2_rebuild",
                "method": c["method"],
                "score": round(c["mean_01"], 4),
                "sub_scores": {k: round(v, 4) for k, v in c["subs"].items()},
                "task_name": c["task_name"],
                "evalverse_ref_path": c["evalverse_ref_path"],
                "prompt_zh": c["prompt_zh"],
            },
        })
    return out


def collect_train_urls(prompt_pair_path):
    if not os.path.exists(prompt_pair_path):
        return set()
    d = json.load(open(prompt_pair_path))
    urls = set()
    for p in d:
        resp = p.get("response")
        if isinstance(resp, list):
            for r in resp:
                if isinstance(r, str):
                    urls.add(r)
                elif isinstance(r, dict):
                    for v in r.values():
                        if isinstance(v, str) and v.startswith("http"):
                            urls.add(v)
        elif isinstance(resp, str):
            urls.add(resp)
        md = p.get("metadata", {}) or {}
        for k in ("winner_url", "loser_url", "url_winner", "url_loser"):
            v = md.get(k)
            if isinstance(v, str):
                urls.add(v)
    return urls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", default="chromaticity",
                    choices=list(DIM_SUB_KEYS.keys()))
    ap.add_argument("--target_count", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workspace_root", default=".",
                    help="repo root containing merged_evaluate_results_with_path_part*.jsonl")
    ap.add_argument("--prompt_pair",
                    default=None,
                    help="path to v2_prompt_pair.json for train URL exclusion. "
                         "default: data/v_arr_mvp/<dim>_mvp/v2_prompt_pair.json")
    ap.add_argument("--out", default=None,
                    help="output json path. default: "
                         "data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>_v2.json")
    args = ap.parse_args()

    root = Path(args.workspace_root).resolve()
    jsonl_paths = sorted(root.glob("merged_evaluate_results_with_path_part*.jsonl"))
    if not jsonl_paths:
        sys.exit(f"[err] No part files found under {root}")
    print(f"[info] candidate sources: {len(jsonl_paths)} part files")

    pp_path = args.prompt_pair or str(root / f"data/v_arr_mvp/{args.dim}_mvp/v2_prompt_pair.json")
    train_urls = collect_train_urls(pp_path)
    print(f"[info] train URLs to exclude: {len(train_urls)} (from {pp_path})")

    sub_keys = DIM_SUB_KEYS[args.dim]
    candidates, stats = collect_candidates(jsonl_paths, sub_keys, train_urls)
    print(f"[info] collected candidates: {len(candidates)}  stats={dict(stats)}")

    # Show pool distribution
    bands_counts = Counter()
    for c in candidates:
        l = c["label"]
        if l < 1.8:
            bands_counts["[1.0,1.8)"] += 1
        elif l < 2.6:
            bands_counts["[1.8,2.6)"] += 1
        elif l < 3.4:
            bands_counts["[2.6,3.4)"] += 1
        elif l < 4.2:
            bands_counts["[3.4,4.2)"] += 1
        else:
            bands_counts["[4.2,5.0]"] += 1
    print(f"[info] candidate band distribution: {dict(bands_counts)}")

    chosen, final_per_band, pool_sizes = stratified_sample(
        candidates, args.target_count, args.seed
    )
    print(f"[info] sampled {len(chosen)} -> per-band={final_per_band} (pool sizes={pool_sizes})")

    labels = [c["label"] for c in chosen]
    if labels:
        label_std = statistics.stdev(labels) if len(labels) >= 2 else 0
        print(f"[info] new test label  min={min(labels):.2f} max={max(labels):.2f}"
              f"  mean={statistics.mean(labels):.2f}  std={label_std:.2f}")

    payload = make_payload(chosen, args.dim, id_prefix=f"varr_{args.dim}_test_v2")

    out_path = args.out or str(root / f"data/v_arr_mvp/{args.dim}_mvp/test_pointwise_{args.dim}_v2.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {out_path} ({len(payload)} samples)")

    # Manifest
    manifest = {
        "dim": args.dim,
        "target_count": args.target_count,
        "n_candidates": len(candidates),
        "candidate_band_distribution": dict(bands_counts),
        "final_per_band": final_per_band,
        "label_std": round(statistics.stdev(labels), 4) if len(labels) >= 2 else 0,
        "label_mean": round(statistics.mean(labels), 4) if labels else 0,
        "label_min": round(min(labels), 2) if labels else None,
        "label_max": round(max(labels), 2) if labels else None,
        "n_train_urls_excluded": len(train_urls),
        "stats": dict(stats),
        "seed": args.seed,
        "source_jsonl": [str(p) for p in jsonl_paths],
    }
    mf_path = out_path.replace(".json", ".manifest.json")
    with open(mf_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[ok] wrote {mf_path}")


if __name__ == "__main__":
    main()

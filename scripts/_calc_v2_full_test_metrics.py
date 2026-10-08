"""Compute per-dim + macro-avg + pooled PLCC/SRCC/MAE/bias for a
scalar_drift per_pointwise.jsonl file.

Usage:
    python3 scripts/_calc_v2_full_test_metrics.py \
        runs/scalar_drift/v2_full_test/per_pointwise.jsonl
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    d = math.sqrt(vx * vy)
    return cov / d if d > 0 else float("nan")


def spearman(xs, ys):
    def avg_rank(vs):
        n = len(vs)
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    return pearson(avg_rank(xs), avg_rank(ys))


def zscore(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, n - 1))
    return [(x - m) / sd for x in xs] if sd > 1e-9 else [0.0] * n


def main():
    if len(sys.argv) < 2:
        print("usage: _calc_v2_full_test_metrics.py <per_pointwise.jsonl>",
              file=sys.stderr)
        sys.exit(2)
    fp = Path(sys.argv[1])
    if not fp.exists():
        print(f"[fatal] not found: {fp}", file=sys.stderr)
        sys.exit(2)

    by_dim = defaultdict(list)
    n_attempts = 0
    n_parse_fail = 0
    with fp.open() as f:
        for line in f:
            r = json.loads(line)
            n_attempts += 1
            if r["pred"] is None:
                n_parse_fail += 1
                continue
            by_dim[r["dim"]].append((float(r["pred"]), float(r["label"])))

    header = f"{'dim':<18} {'n':>4} {'plcc':>7} {'srcc':>7} {'mae_raw':>8} {'mae_zs':>7} {'bias':>7}"
    print(header)
    print("-" * len(header))

    plccs, srccs, maes_raw, maes_zs, biases, ns = [], [], [], [], [], []
    for dim in sorted(by_dim):
        xs = [p for p, _ in by_dim[dim]]
        ys = [l for _, l in by_dim[dim]]
        if len(xs) < 2:
            continue
        p = pearson(xs, ys)
        s = spearman(xs, ys)
        mae = sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)
        zp, zl = zscore(xs), zscore(ys)
        mae_z = sum(abs(a - b) for a, b in zip(zp, zl)) / len(xs)
        bias = sum(xs) / len(xs) - sum(ys) / len(ys)
        print(f"{dim:<18} {len(xs):>4d} {p:>+7.3f} {s:>+7.3f} "
              f"{mae:>8.3f} {mae_z:>7.3f} {bias:>+7.3f}")
        plccs.append(p)
        srccs.append(s)
        maes_raw.append(mae)
        maes_zs.append(mae_z)
        biases.append(bias)
        ns.append(len(xs))

    n_d = len(plccs)
    print("-" * len(header))
    if n_d:
        print(f"{'MACRO_AVG':<18} {sum(ns):>4d} "
              f"{sum(plccs) / n_d:>+7.3f} {sum(srccs) / n_d:>+7.3f} "
              f"{sum(maes_raw) / n_d:>8.3f} {sum(maes_zs) / n_d:>7.3f} "
              f"{sum(biases) / n_d:>+7.3f}")

    all_xs = [x for lst in by_dim.values() for x, _ in lst]
    all_ys = [y for lst in by_dim.values() for _, y in lst]
    if all_xs:
        zp, zl = zscore(all_xs), zscore(all_ys)
        print(f"{'POOLED':<18} {len(all_xs):>4d} "
              f"{pearson(all_xs, all_ys):>+7.3f} "
              f"{spearman(all_xs, all_ys):>+7.3f} "
              f"{sum(abs(x - y) for x, y in zip(all_xs, all_ys)) / len(all_xs):>8.3f} "
              f"{sum(abs(a - b) for a, b in zip(zp, zl)) / len(all_xs):>7.3f} "
              f"{sum(all_xs) / len(all_xs) - sum(all_ys) / len(all_ys):>+7.3f}")

    rate = n_parse_fail / n_attempts if n_attempts else 0.0
    print(f"\nn_attempts={n_attempts}  n_parse_fail={n_parse_fail}  "
          f"parse_fail_rate={rate:.4f}")


if __name__ == "__main__":
    main()

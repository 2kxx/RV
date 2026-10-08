"""Score every video in a manifest under multiple reward models, sequentially.

Rationale
---------
Reward models don't co-fit on a single 80GB GPU (RewardVerse's Qwen2.5-VL-7B
alone is ~16GB, plus VideoScore-v2's Mantis-VL, plus VideoAlign's own VLM).
We spawn one ``reward_server`` subprocess at a time, score the full manifest
under that RM, tear it down, then move to the next.

Output
------
Append-only ``scores.jsonl`` with one row per (video, rm) pair:
    {arm, prompt_id, dimension, prompt, sample_idx, video_path, rm_name, score}

Idempotent: on restart, existing (arm, prompt_id, sample_idx, rm_name) rows
are skipped so we can resume after a killed process.

Usage
-----
    python -m scripts.downstream_grpo.eval.score_matrix \\
        --manifest runs/downstream/eval_videos/manifest.jsonl \\
        --out      runs/downstream/eval_scores \\
        --rms      rewardverse,v3_zeroshot,videoscore_v2,videoalign
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

from scripts.downstream_grpo.src.reward_client import RewardClient


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _load_existing_scores(path: Path) -> set[tuple[str, str, int, str]]:
    """Return set of (arm, prompt_id, sample_idx, rm_name) already scored."""
    if not path.exists():
        return set()
    seen: set[tuple[str, str, int, str]] = set()
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        seen.add((r["arm"], r["prompt_id"], int(r["sample_idx"]), r["rm_name"]))
    return seen


def _append_scores(path: Path, rows: Iterable[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _wait_for_socket(sock_path: Path, proc: subprocess.Popen, timeout_s: float = 300.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sock_path.exists():
            return True
        if proc.poll() is not None:
            # server exited before creating the socket
            return False
        time.sleep(1.0)
    return False


def _score_under_rm(
    rm_name: str,
    manifest_rows: list[dict],
    already_scored: set[tuple[str, str, int, str]],
    scores_path: Path,
    log_dir: Path,
    batch_size: int,
    gpu_ordinal: int,
) -> tuple[int, int]:
    """Return (n_new_scored, n_skipped) for this RM."""
    pending = [
        r for r in manifest_rows
        if (r["arm"], r["prompt_id"], int(r["sample_idx"]), rm_name) not in already_scored
    ]
    if not pending:
        print(f"[score_matrix] rm={rm_name} all manifest rows already scored; skip.")
        return 0, len(manifest_rows)

    sock_path = Path(f"/tmp/eval_{rm_name}.sock")
    if sock_path.exists():
        sock_path.unlink()

    log_path = log_dir / f"reward_server_{rm_name}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_ordinal)

    print(
        f"[score_matrix] rm={rm_name} pending={len(pending)}/{len(manifest_rows)} "
        f"gpu={gpu_ordinal} sock={sock_path} log={log_path}"
    )
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "scripts.downstream_grpo.src.reward_server",
                "--rm_type", rm_name,
                "--socket", str(sock_path),
            ],
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )

    try:
        if not _wait_for_socket(sock_path, proc, timeout_s=600):
            print(f"[score_matrix] rm={rm_name} SERVER FAILED TO START. See {log_path}. Skipping.")
            if proc.poll() is None:
                proc.kill()
            return 0, len(pending)

        client = RewardClient(str(sock_path), connect_timeout=60.0)

        n_scored = 0
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            batch = [
                {
                    "prompt": r["prompt"],
                    "dimension": r["dimension"],
                    "video_path": r["video_path"],
                }
                for r in chunk
            ]
            try:
                scores = client.score(batch).tolist()
            except Exception as exc:  # noqa: BLE001
                print(f"[score_matrix] rm={rm_name} batch@{start} failed: {exc}; skipping chunk.")
                continue

            _append_scores(scores_path, [
                {
                    "arm": r["arm"],
                    "prompt_id": r["prompt_id"],
                    "dimension": r["dimension"],
                    "prompt": r["prompt"],
                    "sample_idx": int(r["sample_idx"]),
                    "video_path": r["video_path"],
                    "rm_name": rm_name,
                    "score": float(s),
                }
                for r, s in zip(chunk, scores)
            ])
            n_scored += len(chunk)
            if (start // batch_size) % 5 == 0:
                print(f"[score_matrix]   rm={rm_name} {n_scored}/{len(pending)}")
        return n_scored, len(pending) - n_scored
    finally:
        # graceful shutdown, then hard kill if it hangs
        try:
            RewardClient(str(sock_path), connect_timeout=5.0).shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if sock_path.exists():
            try:
                sock_path.unlink()
            except OSError:
                pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True, help="Directory for scores.jsonl and server logs")
    ap.add_argument("--rms", default="rewardverse,v3_zeroshot,videoscore_v2,videoalign",
                    help="Comma-separated RM adapter names (see reward_adapters._REGISTRY)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--gpu_ordinal", type=int, default=0,
                    help="Physical GPU ID pinned to the reward server subprocess.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    rows = _load_manifest(manifest_path)
    if not rows:
        print(f"[score_matrix] manifest {manifest_path} is empty; nothing to score.")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"
    log_dir = out_dir / "server_logs"

    rm_names = [r.strip() for r in args.rms.split(",") if r.strip()]
    print(f"[score_matrix] manifest_rows={len(rows)} rms={rm_names}")
    print(f"[score_matrix] scores -> {scores_path}")

    already_scored = _load_existing_scores(scores_path)

    totals: dict[str, tuple[int, int]] = {}
    for rm in rm_names:
        try:
            new, skipped = _score_under_rm(
                rm_name=rm,
                manifest_rows=rows,
                already_scored=already_scored,
                scores_path=scores_path,
                log_dir=log_dir,
                batch_size=args.batch_size,
                gpu_ordinal=args.gpu_ordinal,
            )
            totals[rm] = (new, skipped)
            # refresh the cache so re-runs of the same RM in one process don't double-score
            already_scored = _load_existing_scores(scores_path)
        except KeyboardInterrupt:
            print("[score_matrix] interrupted by user; partial scores retained.")
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[score_matrix] rm={rm} UNHANDLED ERROR: {exc}; skipping.")
            totals[rm] = (0, 0)

    print("[score_matrix] summary:")
    for rm, (new, skipped) in totals.items():
        print(f"  {rm}: scored {new} new, {skipped} skipped")
    print(f"[score_matrix] done -> {scores_path}")


if __name__ == "__main__":
    main()

"""Standalone reward server: one process holds one reward model
on a dedicated GPU and serves requests via a Unix domain socket.

Protocol (line-delimited JSON):
    C -> S: {"batch": [{"prompt": ..., "dimension": ..., "video_path": ...}, ...]}
    S -> C: {"rewards": [float, ...]}   or   {"error": "..."}

Launch (see launch/run_arm_*.sh):
    CUDA_VISIBLE_DEVICES=7 python -m scripts.downstream_grpo.src.reward_server \\
        --rm_type rewardverse \\
        --socket /tmp/rewardverse_rm.sock

Why a separate process
----------------------
* Isolates RM memory (Qwen2.5-VL-7B ≈ 16 GB, VideoScore ≈ 5 GB) from
  the diffusion trainer's DeepSpeed state.
* Swap reward models with a single CLI flag.
* Keeps async Judger calls off the trainer's event loop.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

import torch

from scripts.downstream_grpo.src.reward_adapters import RewardRequest, build_adapter


def _recv_line(conn: socket.socket) -> str:
    buf = bytearray()
    while True:
        chunk = conn.recv(65536)
        if not chunk:
            break
        buf.extend(chunk)
        if b"\n" in buf:
            break
    line, _, _ = buf.partition(b"\n")
    return line.decode("utf-8")


def _send_line(conn: socket.socket, payload: str) -> None:
    conn.sendall((payload + "\n").encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm_type", required=True, help="Registered adapter name")
    ap.add_argument("--socket", required=True, help="Unix socket path")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in reward server")

    print(f"[reward_server] loading adapter '{args.rm_type}' on {device}...", flush=True)
    adapter = build_adapter(args.rm_type, device=device)
    print(f"[reward_server] adapter '{adapter.name}' ready.", flush=True)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(4)
    print(f"[reward_server] listening on {sock_path}", flush=True)

    try:
        while True:
            conn, _ = server.accept()
            try:
                line = _recv_line(conn)
                if not line:
                    continue
                req = json.loads(line)
                if req.get("op") == "shutdown":
                    _send_line(conn, json.dumps({"ok": True}))
                    break
                batch = [
                    RewardRequest(
                        prompt=b["prompt"],
                        dimension=b["dimension"],
                        video_path=b["video_path"],
                    )
                    for b in req["batch"]
                ]
                scores = adapter.score_batch(batch)
                _send_line(conn, json.dumps({"rewards": scores.tolist()}))
            except Exception as exc:  # noqa: BLE001
                _send_line(conn, json.dumps({"error": str(exc)}))
            finally:
                conn.close()
    finally:
        adapter.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())

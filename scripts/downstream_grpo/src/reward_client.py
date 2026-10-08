"""Small client that talks to reward_server.py over a Unix socket."""

from __future__ import annotations

import json
import socket
import time
from typing import Sequence

import torch


class RewardClient:
    def __init__(self, socket_path: str, connect_timeout: float = 300.0) -> None:
        self.socket_path = socket_path
        deadline = time.time() + connect_timeout
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(socket_path)
                s.close()
                return
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(1.0)
        raise TimeoutError(f"reward server not up at {socket_path}")

    def score(self, batch: Sequence[dict]) -> torch.Tensor:
        """batch: list of {"prompt", "dimension", "video_path"} dicts."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(self.socket_path)
        payload = json.dumps({"batch": list(batch)}) + "\n"
        s.sendall(payload.encode("utf-8"))
        buf = bytearray()
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if b"\n" in buf:
                break
        s.close()
        line, _, _ = buf.partition(b"\n")
        resp = json.loads(line.decode("utf-8"))
        if "error" in resp:
            raise RuntimeError(f"reward server error: {resp['error']}")
        return torch.tensor(resp["rewards"], dtype=torch.float32)

    def shutdown(self) -> None:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self.socket_path)
            s.sendall((json.dumps({"op": "shutdown"}) + "\n").encode("utf-8"))
            s.close()
        except Exception:  # noqa: BLE001
            pass

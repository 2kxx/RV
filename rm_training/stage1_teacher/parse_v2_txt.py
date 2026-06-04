"""rm_training.stage1_teacher.parse_v2_txt
=======================================
Parse the per-dimension rubric files emitted by the offline teacher run::

    rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt
    rubric_pipeline/rubrics/v_arr_mvp_vq_v2_prompt_pair.txt   # VQ uses 'vq'

Each file is a plain-text scaffold (no JSON, no weights) with K=5 entries::

    Rubric 1:
    Theme: <theme line>
    - Tip1: ...
    - Tip2: ...
    - Tip3: ...
    [- Tip4: ...]                                # some dims use 4 tips

    Rubric 2:
    Theme: ...
    ...

The Stage ② training loop only consumes the *path* (Theme line) for the
gold-align bipartite matching; it never reads weights or tips. We still
attach an even-weight default (``1/K``) plus the parsed tips so future
ablations (weighted gold-align, NLI-style tip matching) have the data
without re-parsing.

Public API
----------
- ``parse_rubric_text(text) -> list[dict]``
- ``parse_rubric_file(path) -> list[dict]``
- ``DIM_TO_RUBRIC_FILENAME`` — dim → relative filename inside rubrics/
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "parse_rubric_text",
    "parse_rubric_file",
    "DIM_TO_RUBRIC_FILENAME",
    "discover_rubric_dir",
]


# Most dims follow ``v_arr_mvp_<dim>_v2.txt``; visual_quality is the
# exception — its file name is ``v_arr_mvp_vq_v2_prompt_pair.txt`` (the
# original teacher run was tagged with the VGRB axis label "vq").
DIM_TO_RUBRIC_FILENAME: dict[str, str] = {
    "visual_quality": "v_arr_mvp_vq_v2_prompt_pair.txt",
    # Anything not in this map falls back to v_arr_mvp_<dim>_v2.txt.
}


def _filename_for(dim: str) -> str:
    return DIM_TO_RUBRIC_FILENAME.get(dim, f"v_arr_mvp_{dim}_v2.txt")


def discover_rubric_dir(repo_root: Path | str) -> Path:
    """Return the absolute path to ``rubric_pipeline/rubrics/`` under ``repo_root``."""
    return Path(repo_root) / "rubric_pipeline" / "rubrics"


# ---------------------------------------------------------------------------
# Parser.
# ---------------------------------------------------------------------------
_RUBRIC_HEADER_RE = re.compile(r"^Rubric\s+(\d+)\s*:\s*$", re.MULTILINE)
_THEME_RE        = re.compile(r"^Theme\s*:\s*(.+?)\s*$", re.MULTILINE)
_TIP_RE          = re.compile(r"^\s*-\s*Tip\s*\d*\s*:\s*(.+?)\s*$", re.MULTILINE)


def parse_rubric_text(text: str, *, expected_k: int = 5) -> list[dict]:
    """Parse a single ``v2.txt`` blob into ``[{path, weight, tips}, ...]``.

    Schema returned per entry::

        {
          "path":   "Theme: <theme line>",   # ← used by gold-align cosine
          "weight": 1.0 / K,                 # equal weights (txt has none)
          "tips":   ["<tip1>", "<tip2>", ...]
        }

    Why the path keeps the literal ``"Theme: "`` prefix
        Stage ② code (``rubric_parser._EXPECTED_K`` and the dummy student's
        ``_DUMMY_PATHS``) all use the ``"Theme: ..."`` convention. Keeping
        it here means cosine matching against student rubrics works without
        a normalisation step.
    """
    if not text:
        return []

    # Split into chunks at every "Rubric N:" header. We intentionally keep
    # the splitter's text out of the chunks so the regexes below match the
    # remaining body deterministically.
    headers = list(_RUBRIC_HEADER_RE.finditer(text))
    if not headers:
        return []

    chunks: list[str] = []
    for i, h in enumerate(headers):
        start = h.end()
        end   = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        chunks.append(text[start:end])

    out: list[dict] = []
    for chunk in chunks:
        theme_m = _THEME_RE.search(chunk)
        if not theme_m:
            # Skip malformed rubric block; never raise — Stage ② treats
            # missing rubrics as low r_align rather than crash.
            continue
        theme  = theme_m.group(1).strip()
        path   = f"Theme: {theme}" if not theme.lower().startswith("theme") else theme
        tips   = [m.group(1).strip() for m in _TIP_RE.finditer(chunk)]
        out.append({
            "path":   path,
            "weight": 0.0,        # filled in below after we know K
            "tips":   tips,
        })

    if not out:
        return []

    # Equal-weight: 1/K. Compensate rounding drift on the last entry so
    # Σ weights == 1.0 exactly (the format-reward check uses ±0.05 anyway).
    k = len(out)
    w = round(1.0 / k, 6)
    for r in out:
        r["weight"] = w
    out[-1]["weight"] = round(1.0 - w * (k - 1), 6)

    if expected_k and k != expected_k:
        # Soft warning via a sentinel field; the caller logs and continues.
        out_meta = {"_unexpected_k": k}
        out.append({"_meta": out_meta, "path": "", "weight": 0.0, "tips": []})
        # Strip the sentinel before returning — callers should observe `len()`.
        out.pop()
    return out


def parse_rubric_file(path: str | Path, *, expected_k: int = 5) -> list[dict]:
    """Read + parse one ``v2.txt`` file. Returns [] on missing / unreadable file."""
    p = Path(path)
    if not p.is_file():
        return []
    return parse_rubric_text(p.read_text(encoding="utf-8"), expected_k=expected_k)

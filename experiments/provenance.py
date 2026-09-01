"""Run manifests and per-trajectory artifacts for live experiments."""
from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import json
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
_SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "results"}


def _git(*args: str) -> str | None:
    try:
        p = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=10)
    except OSError:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def git_is_dirty() -> bool:
    """True when tracked or untracked files differ from the recorded commit."""
    return bool(_git("status", "--porcelain"))


def _endpoint_family(raw: str | None) -> str | None:
    """Record endpoint provenance without userinfo, paths, queries or secrets."""
    if not raw:
        return None
    parsed = urlsplit(raw)
    return f"{parsed.scheme}://{parsed.hostname}" if parsed.hostname else "configured"


def run_manifest(args: argparse.Namespace | dict[str, Any]) -> dict[str, Any]:
    """Everything needed to identify the code and command behind a result."""
    values = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    packages = {}
    for name in ("openai", "anthropic", "pytest", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": git_is_dirty(),
        "command": [sys.executable, *sys.argv],
        "args": values,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        # Record which endpoint family was used without ever persisting keys.
        "openai_endpoint": _endpoint_family(os.environ.get("OPENAI_BASE_URL")),
    }


def randomized_block_schedule(blocks, treatments, *, seed: int):
    """Shuffle blocks and treatment order within every block reproducibly."""
    blocks = list(blocks)
    treatments = list(treatments)
    rng = random.Random(seed)
    rng.shuffle(blocks)
    out = []
    for block in blocks:
        cells = list(treatments)
        rng.shuffle(cells)
        out.extend((cell, block) for cell in cells)
    return out


def snapshot_text(root: str | Path, *, max_bytes: int = 1_000_000) -> dict[str, str]:
    """Capture small text files so a final unified patch can be reconstructed."""
    root = Path(root)
    out: dict[str, str] = {}
    for p in root.rglob("*"):
        if not p.is_file() or any(part in _SKIP_PARTS for part in p.parts):
            continue
        try:
            if p.stat().st_size > max_bytes:
                continue
            out[p.relative_to(root).as_posix()] = p.read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def unified_patch(before: dict[str, str], root: str | Path) -> str:
    after = snapshot_text(root)
    chunks: list[str] = []
    for rel in sorted(set(before) | set(after)):
        a = before.get(rel, "").splitlines(keepends=True)
        b = after.get(rel, "").splitlines(keepends=True)
        if a == b:
            continue
        chunks.extend(difflib.unified_diff(
            a, b, fromfile=f"a/{rel}", tofile=f"b/{rel}", lineterm="\n"))
    return "".join(chunks)


def save_artifact(directory: str | Path, artifact: dict[str, Any]) -> Path:
    """Write one trajectory per file; concurrent workers never append together."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trace = artifact.get("trace") or {}
    raw = str(trace.get("run_id") or artifact.get("run_key") or "run")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)[:100] + ".json"
    path = directory / name
    path.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")
    return path

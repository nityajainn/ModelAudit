"""Small provenance helpers for local, non-overwriting experiment outputs."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess


def fingerprint(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: dict) -> None:
    # Serialize before creating a file so invalid JSON cannot leave a partial file.
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def run_context() -> dict:
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    try:
        def git(*args):
            return subprocess.check_output(
                ["git", *args], cwd=root, env=env, text=True,
                stderr=subprocess.DEVNULL, timeout=10,
            ).strip()

        git_state = {"commit": git("rev-parse", "HEAD"),
                     "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        git_state = {"commit": None, "dirty": None}
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in
                     ("modelaudit", "numpy", "pandas", "scikit-learn", "joblib", "scipy")},
        "git": git_state,
    }

"""Read-only storage preflight. Never run simulations, stage files, or delete data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess


GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]


def nonnegative(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}, check=False,
    )


def git_output(*args: str) -> bytes:
    result = git(*args)
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout


def git_paths(*args: str) -> list[str]:
    return [os.fsdecode(path) for path in git_output(*args).split(b"\0") if path]


def inventory(root: Path, *, skip_git: bool = False) -> tuple[int, list[str]]:
    """Count logical file bytes, never following links/junctions out of the tree."""
    total = 0
    links = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if skip_git and directory == root and entry.name == ".git":
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue  # A runtime cache may disappear during enumeration.
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
                ):
                    links.append(str(Path(entry.path).relative_to(root)))
                elif stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
    return total, links


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--expected-growth-gib", type=nonnegative, default=0)
    parser.add_argument("--max-git-gib", type=nonnegative, default=1)
    parser.add_argument("--max-project-gib", type=nonnegative, default=5)
    parser.add_argument("--min-free-gib", type=nonnegative, default=10)
    args = parser.parse_args()
    failures = []
    common = Path(os.fsdecode(git_output("rev-parse", "--path-format=absolute", "--git-common-dir")).strip())
    git_bytes, git_links = inventory(common)
    working_bytes, working_links = inventory(ROOT, skip_git=True)
    project_bytes = working_bytes + git_bytes
    free_bytes = shutil.disk_usage(ROOT).free
    growth = math.ceil(args.expected_growth_gib * GIB)
    if git_bytes > args.max_git_gib * GIB:
        failures.append("Git size exceeds budget; inspect growth before launching.")
    if project_bytes + growth > args.max_project_gib * GIB:
        failures.append("Project plus planned growth exceeds budget.")
    if free_bytes - growth < args.min_free_gib * GIB:
        failures.append("Insufficient free-space reserve after planned growth.")
    if git_links or working_links:
        failures.append("Links/reparse points were excluded from the byte count; review storage manually.")

    tracked_ignored = git_paths("ls-files", "--cached", "--ignored", "--exclude-standard", "-z")
    if tracked_ignored:
        failures.append("Ignored files are still tracked; review index entries without deleting working files.")

    artifact_dir = None
    if args.artifact_dir is not None:
        artifact_dir = (ROOT / args.artifact_dir).resolve()
        try:
            relative = artifact_dir.relative_to(ROOT)
        except ValueError:
            failures.append("Artifact directory must be inside this repository's _artifacts/<study-id>/.")
        else:
            if len(relative.parts) < 2 or relative.parts[0] != "_artifacts":
                failures.append("Artifact directory must be _artifacts/<study-id>/.")
            elif artifact_dir.exists() and not artifact_dir.is_dir():
                failures.append("Artifact directory points to a file.")
            else:
                probe = (relative / "__storage_ignore_probe__.npz").as_posix()
                result = git("check-ignore", "-q", "--", probe)
                if result.returncode != 0:
                    failures.append("Artifact directory is not effectively ignored by Git.")

    large_visible = []
    for name in set(git_paths("ls-files", "--cached", "--others", "--exclude-standard", "-z")):
        path = ROOT / name
        if path.is_file() and not path.is_symlink():
            size = path.stat().st_size
            if size > 50 * 1024 ** 2:
                large_visible.append({"path": name, "bytes": size})
    report = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if failures else "PASS",
        "root": str(ROOT), "git_common_dir": str(common),
        "git_bytes": git_bytes, "working_file_bytes": working_bytes,
        "project_bytes": project_bytes, "free_bytes": free_bytes,
        "expected_growth_bytes": growth,
        "limits_gib": {"git": args.max_git_gib, "project": args.max_project_gib, "free_reserve": args.min_free_gib},
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
        "tracked_ignored_count": len(tracked_ignored),
        "tracked_ignored_examples": tracked_ignored[:10],
        "excluded_link_count": len(git_links) + len(working_links),
        "large_git_visible_files": sorted(large_visible, key=lambda item: item["bytes"], reverse=True)[:10],
        "failures": failures,
        "scope": "Preflight only; no in-run quota, monitoring, output redirection, or deletion.",
    }
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as exc:
        print(json.dumps({"status": "ERROR", "message": str(exc)}))
        raise SystemExit(2)

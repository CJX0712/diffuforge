"""Three-level fallback push of a local repo to GitHub.

Usage::

    python scripts/gh_push.py \\
        --repo diffuforge \\
        --local . \\
        --description "NFE-budgeted diffusion / score-based modeling with the DiffuFuse sampler" \\
        --tag v0.1.0 --release

Levels (each tried in order; the first that works wins):
  1. ``gh repo create --source . --remote origin --push``  (create + push at once)
  2. ``gh repo create`` (no push) + ``git push -u origin <branch>``
  3. ``gh api repos`` (POST) to create + ``git push``

If the repo already exists on GitHub, the create step is skipped and we go straight to
pushing.  The script also makes sure the local tree is a git repo with a commit, creates the
tag, and optionally drafts a GitHub Release.

Author: 晨星 (CJX0712).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def _owner() -> str:
    r = _run(["gh", "api", "user", "--jq", ".login"])
    if r.returncode != 0 or not r.stdout.strip():
        raise SystemExit("cannot determine GitHub owner (is `gh` authenticated?)")
    return r.stdout.strip()


def _repo_exists(owner: str, repo: str) -> bool:
    r = _run(["gh", "repo", "view", f"{owner}/{repo}"])
    return r.returncode == 0


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return _run(["git", *args], cwd=cwd)


def _ensure_local_commit(cwd: Path) -> str:
    if not (cwd / ".git").exists():
        _git("init", "-q", cwd=cwd)
    # make sure there is a branch name (Windows: default may be "master")
    branch_proc = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)
    branch = branch_proc.stdout.strip() or "main"
    if branch == "HEAD":  # detached / unborn
        _git("checkout", "-q", "-B", "main", cwd=cwd)
        branch = "main"
    _git("add", "-A", cwd=cwd)
    status = _git("status", "--porcelain", cwd=cwd)
    if status.stdout.strip():
        _git("commit", "-q", "-m", "Initial DiffuForge release (author: 晨星)", cwd=cwd)
    return branch


def _ensure_remote(owner: str, repo: str, cwd: Path) -> None:
    rem = _git("remote", "get-url", "origin", cwd=cwd)
    expected = f"https://github.com/{owner}/{repo}.git"
    if rem.returncode != 0:
        _git("remote", "add", "origin", expected, cwd=cwd)
    elif expected not in rem.stdout:
        _git("remote", "set-url", "origin", expected, cwd=cwd)


def _create_and_push_l1(owner: str, repo: str, desc: str, cwd: Path, branch: str) -> bool:
    r = _run(
        [
            "gh",
            "repo",
            "create",
            f"{owner}/{repo}",
            "--public",
            "--description",
            desc,
            "--source",
            ".",
            "--remote",
            "origin",
            "--push",
        ],
        cwd=cwd,
    )
    if r.returncode == 0:
        print(f"[L1] created + pushed {owner}/{repo}")
        return True
    print(f"[L1] failed: {r.stderr.strip() or r.stdout.strip()}")
    return False


def _create_l2(owner: str, repo: str, desc: str, cwd: Path) -> bool:
    r = _run(
        ["gh", "repo", "create", f"{owner}/{repo}", "--public", "--description", desc],
        cwd=cwd,
    )
    if r.returncode == 0:
        print(f"[L2] created {owner}/{repo} (no push)")
        return True
    print(f"[L2] failed: {r.stderr.strip() or r.stdout.strip()}")
    return False


def _create_l3(owner: str, repo: str, desc: str) -> bool:
    r = _run(
        [
            "gh",
            "api",
            "repos",
            "-f",
            f"name={repo}",
            "-f",
            f"description={desc}",
            "-f",
            "private=false",
        ]
    )
    if r.returncode == 0:
        print("[L3] created repo via gh api")
        return True
    print(f"[L3] failed: {r.stderr.strip() or r.stdout.strip()}")
    return False


def _push(owner: str, repo: str, cwd: Path, branch: str) -> bool:
    _ensure_remote(owner, repo, cwd)
    r = _git("push", "-u", "origin", branch, cwd=cwd)
    if r.returncode == 0:
        print(f"[push] pushed {branch} -> origin")
        return True
    print(f"[push] failed: {r.stderr.strip() or r.stdout.strip()}")
    # retry once without -u (e.g. branch already tracked)
    r2 = _git("push", "origin", branch, cwd=cwd)
    if r2.returncode == 0:
        print(f"[push] pushed {branch} -> origin (retry)")
        return True
    print(f"[push] retry failed: {r2.stderr.strip() or r2.stdout.strip()}")
    return False


def _tag_and_release(owner: str, repo: str, tag: str, cwd: Path, make_release: bool) -> None:
    existing = _git("tag", "--list", tag, cwd=cwd)
    if not existing.stdout.strip():
        _git("tag", "-a", tag, "-m", f"DiffuForge {tag} — author 晨星", cwd=cwd)
        print(f"[tag] created {tag}")
    _git("push", "origin", tag, cwd=cwd)
    if make_release:
        rel = _run(
            [
                "gh",
                "release",
                "create",
                tag,
                "--title",
                f"DiffuForge {tag}",
                "--notes",
                "NFE-budgeted diffusion / score-based generative modeling "
                "with the DiffuFuse sampler. Author: 晨星 (CJX0712). See README.md.",
                "--repo",
                f"{owner}/{repo}",
            ]
        )
        if rel.returncode == 0:
            print(f"[release] created GitHub release {tag}")
        else:
            print(f"[release] skipped/failed: {rel.stderr.strip() or rel.stdout.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Three-level fallback push to GitHub.")
    ap.add_argument("--repo", required=True, help="repository name (e.g. diffuforge)")
    ap.add_argument("--local", default=".", help="local repo path")
    ap.add_argument("--description", default="DiffuForge — NFE-budgeted diffusion samplers")
    ap.add_argument("--tag", default="", help="git tag to create/push (e.g. v0.1.0)")
    ap.add_argument("--release", action="store_true", help="create a GitHub Release")
    args = ap.parse_args()

    cwd = Path(args.local).resolve()
    owner = _owner()
    branch = _ensure_local_commit(cwd)

    # If the repo already exists on GitHub, skip creation and go straight to push.
    exists = _repo_exists(owner, args.repo)
    if exists:
        print(f"[info] {owner}/{args.repo} already exists — skipping creation")

    pushed = False
    if not exists:
        # Short-circuit: only attempt each level if the previous one failed.
        pushed = (
            _create_and_push_l1(owner, args.repo, args.description, cwd, branch)
            or (
                _create_l2(owner, args.repo, args.description, cwd)
                and _push(owner, args.repo, cwd, branch)
            )
            or (
                _create_l3(owner, args.repo, args.description)
                and _push(owner, args.repo, cwd, branch)
            )
        )
    else:
        pushed = _push(owner, args.repo, cwd, branch)

    if not pushed:
        print("[FATAL] all push levels failed", file=sys.stderr)
        return 1

    if args.tag:
        _tag_and_release(owner, args.repo, args.tag, cwd, args.release)

    print(f"[done] https://github.com/{owner}/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

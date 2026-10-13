#!/usr/bin/env python3
"""Preview, check, commit and fast-forward push SpecCraft code. Python 3.9+."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys


EXPECTED_REMOTE = "https://github.com/Soberheart/SpecCraft.git"
ALLOWED_REMOTES = {
    EXPECTED_REMOTE, EXPECTED_REMOTE[:-4],
    "git@github.com:Soberheart/SpecCraft.git",
    "ssh://git@github.com/Soberheart/SpecCraft.git",
}
MAX_BYTES = 25 * 1024 * 1024
BLOCKED_ROOTS = {
    "data", "weights", "outputs", "records", "results", "wandb", "backups",
    "level4_backups", "level4_restore_checks", "runtime_snapshot", ".ssh",
    "venv", ".venv", "env", "net/nvdiffrast", "sim/sdftoolbox",
}
BLOCKED_SUFFIXES = {
    ".h5", ".hdf5", ".pth", ".pt", ".ckpt", ".safetensors", ".pkl",
    ".npz", ".npy", ".zip", ".tar", ".gz", ".7z", ".mp4", ".gif",
    ".pem", ".key", ".p12", ".pfx", ".log",
}
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"(?i)https?://[^\s/@]+:[^\s/@]+@github\.com"),
)


class SyncError(RuntimeError):
    pass


def git(repo, *args, check=True):
    result = subprocess.run(["git", "-C", str(repo), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        # Do not echo raw Git errors: a misconfigured remote may embed credentials.
        raise SyncError("Git '{}' failed (exit {}). Check authentication, network "
                        "and repository permissions locally.".format(args[0], result.returncode))
    return result


def decode(data):
    return data.decode("utf-8", errors="surrogateescape")


def split_paths(data):
    return [decode(item) for item in data.split(b"\0") if item]


def policy_check(path, content):
    """Return only reasons, never file contents or detected credentials."""
    lowered = path.lower()
    parts = PurePosixPath(lowered).parts
    name = PurePosixPath(lowered).name
    reasons = []
    if any(lowered == root or lowered.startswith(root + "/") for root in BLOCKED_ROOTS):
        reasons.append("local asset/third-party directory")
    if PurePosixPath(lowered).suffix in BLOCKED_SUFFIXES:
        reasons.append("data, checkpoint, archive, log or credential file")
    if (any(part == ".env" or (part.startswith(".env.") and part != ".env.example")
            for part in parts) or name.startswith(("id_rsa", "id_ed25519", "service-account"))
            or name == "credentials.json"):
        reasons.append("credential filename")
    if len(content) > MAX_BYTES:
        reasons.append("larger than 25 MiB; store separately or use an explicit LFS workflow")
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        reasons.append("possible credential in file content")
    return reasons


def ensure_safe(entries):
    problems = []
    for path, content in entries:
        reasons = policy_check(path, content)
        if reasons:
            problems.append("{}: {}".format(path, "; ".join(reasons)))
    if problems:
        raise SyncError("Upload blocked:\n  " + "\n  ".join(problems)
                        + "\nReview these files manually. Ignoring an already tracked file "
                        "does not remove it from Git history.")


def working_entries(repo):
    changed = split_paths(git(repo, "diff", "HEAD", "--no-renames", "--name-only", "-z").stdout)
    untracked = split_paths(git(repo, "ls-files", "--others", "--exclude-standard", "-z").stdout)
    for path in sorted(set(changed + untracked)):
        file_path = repo / path
        if file_path.is_symlink():
            yield path, os.readlink(file_path).encode("utf-8", errors="surrogateescape")
        elif file_path.is_file():
            # Never load a huge dataset into RAM simply to reject it.
            if file_path.stat().st_size > MAX_BYTES:
                raise SyncError("Upload blocked: {} is larger than 25 MiB.".format(path))
            yield path, file_path.read_bytes()
        elif file_path.is_dir():
            raise SyncError("Changed nested repository/directory: {}. Sync separately.".format(path))


def snapshot_entries(repo, revision):
    paths = split_paths(git(repo, "diff-tree", "--root", "--no-commit-id", "-r", "-m",
                            "--no-renames", "--name-only", "-z", "--diff-filter=ACMRT",
                            revision).stdout)
    for path in paths:
        spec = "{}:{}".format(revision, path)
        size = int(git(repo, "cat-file", "-s", spec).stdout)
        if size > MAX_BYTES:
            raise SyncError("Upload blocked: oversized file in outgoing commit: {}".format(path))
        yield path, git(repo, "show", spec).stdout


@contextmanager
def sync_lock(repo):
    lock_path = Path(decode(git(repo, "rev-parse", "--path-format=absolute",
                               "--git-path", "speccraft-sync.lock").stdout).strip())
    try:
        with lock_path.open("x", encoding="ascii") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise SyncError("Another sync may be running. Inspect {} before removing a stale lock.".format(lock_path))
    try:
        yield
    finally:
        lock_path.unlink()


def sync(args):
    if not shutil.which("git"):
        raise SyncError("Install Git and make it available on PATH first.")
    repo = Path(decode(git(args.repo, "rev-parse", "--show-toplevel").stdout).strip())
    git(repo, "rev-parse", "--verify", "HEAD")
    branch = decode(git(repo, "symbolic-ref", "--quiet", "--short", "HEAD").stdout).strip()
    for config_key in ("remote.origin.url", "remote.origin.pushurl"):
        values = git(repo, "config", "--get-all", config_key, check=False)
        urls = decode(values.stdout).splitlines()
        if config_key.endswith(".url") and not urls:
            raise SyncError("origin is missing; expected " + EXPECTED_REMOTE)
        if any(url not in ALLOWED_REMOTES for url in urls):
            raise SyncError("Unexpected origin URL/push URL. Expected " + EXPECTED_REMOTE)
    # get-url resolves Git URL rewrites as well as configured remote values.
    for flag in ([], ["--push"]):
        resolved = decode(git(repo, "remote", "get-url", *flag, "--all", "origin").stdout).splitlines()
        if any(url not in ALLOWED_REMOTES for url in resolved):
            raise SyncError("Git URL rewrite points away from SpecCraft; inspect Git configuration.")
    if git(repo, "config", "--bool", "--get", "remote.origin.mirror", check=False).stdout.strip() == b"true":
        raise SyncError("Mirror push configuration is not allowed.")
    if git(repo, "ls-files", "-u").stdout:
        raise SyncError("Resolve unmerged files before syncing.")
    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply"):
        marker_path = Path(decode(git(repo, "rev-parse", "--path-format=absolute",
                                     "--git-path", marker).stdout).strip())
        if marker_path.exists():
            raise SyncError("Finish or cancel the in-progress Git operation first: " + marker)
    print("SpecCraft repository:", repo)
    print("Destination:", EXPECTED_REMOTE, "| Branch:", branch)
    print(decode(git(repo, "-c", "core.quotePath=false", "status", "--short").stdout)
          or "Working tree is clean.")
    ensure_safe(working_entries(repo))
    if args.dry_run:
        print("Preview passed. No staging, commit, fetch or push performed. "
              "Outgoing commit history is checked during the actual sync.")
        return
    if not args.yes and input("Commit ALL non-ignored changes and push this branch? [y/N] ").strip().lower() != "y":
        print("Cancelled; no changes made.")
        return
    with sync_lock(repo):
        remote_ref = "refs/remotes/origin/" + branch
        git(repo, "fetch", "--no-tags", "origin", "refs/heads/{}:{}".format(branch, remote_ref))
        if git(repo, "merge-base", "--is-ancestor", remote_ref, "HEAD", check=False).returncode:
            raise SyncError("Remote is ahead or histories diverged. No files staged or committed. "
                            "Review git log --oneline --graph --all; use pull --ff-only only "
                            "after saving local changes. This script never merges or force-pushes.")
        # Check already-existing local commits before staging new changes.
        for commit in decode(git(repo, "rev-list", remote_ref + "..HEAD").stdout).splitlines():
            ensure_safe(snapshot_entries(repo, commit))
        ensure_safe(working_entries(repo))
        git(repo, "add", "--all")
        staged_paths = split_paths(git(repo, "diff", "--cached", "--no-renames", "--name-only",
                                       "-z", "--diff-filter=ACMRT").stdout)
        for path in staged_paths:
            if int(git(repo, "cat-file", "-s", ":" + path).stdout) > MAX_BYTES:
                raise SyncError("Oversized staged file: " + path)
            ensure_safe([(path, git(repo, "show", ":" + path).stdout)])
        if git(repo, "diff", "--cached", "--quiet", check=False).returncode:
            message = args.message or "Sync SpecCraft " + datetime.now().astimezone().isoformat(timespec="seconds")
            git(repo, "commit", "-m", message)
            print("Committed:", decode(git(repo, "rev-parse", "--short", "HEAD").stdout).strip())
        else:
            print("No new changes to commit.")
        # Hooks may have modified the commit; check its outgoing content again.
        for commit in decode(git(repo, "rev-list", remote_ref + "..HEAD").stdout).splitlines():
            ensure_safe(snapshot_entries(repo, commit))
        if git(repo, "rev-list", "--count", remote_ref + "..HEAD").stdout.strip() == b"0":
            print("Already up to date.")
            return
        git(repo, "-c", "push.followTags=false", "push", "--porcelain", "--set-upstream",
            "origin", "HEAD:refs/heads/" + branch)
        local_head = git(repo, "rev-parse", "HEAD").stdout.strip()
        remote_head = git(repo, "ls-remote", "--heads", "origin", "refs/heads/" + branch).stdout.split()
        if not remote_head or remote_head[0] != local_head:
            raise SyncError("Push returned but remote HEAD verification differed. Inspect remotely.")
        print("Verified GitHub sync:", decode(local_head), "->", EXPECTED_REMOTE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository path; default is the parent of scripts/.")
    parser.add_argument("--message", help="Commit message; default includes local timestamp.")
    parser.add_argument("--dry-run", action="store_true", help="Read-only local preview, no network writes.")
    parser.add_argument("--yes", action="store_true", help="Explicitly skip the confirmation prompt.")
    args = parser.parse_args()
    try:
        sync(args)
        return 0
    except (SyncError, OSError, EOFError) as error:
        print("Sync stopped:", error, file=sys.stderr)
        print("No force-push/reset performed. If staging or commit already occurred, it is kept "
              "locally for review and retry.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Cancelled; existing local commits and files are preserved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

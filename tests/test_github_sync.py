"""Exercise the sync workflow with temporary local Git repositories, no network."""
import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_github.py"
SPEC = importlib.util.spec_from_file_location("speccraft_sync", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(shutil.which("git"), "Git is required")
class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="speccraft-sync-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "work"
        self.remote = self.root / "remote.git"
        self.repo.mkdir()
        subprocess.run(["git", "init", "--bare", str(self.remote)], check=True, capture_output=True)
        self.original_git = MODULE.git
        self.g("init", "-b", "main")
        self.g("config", "user.name", "Sync Test")
        self.g("config", "user.email", "test@example.invalid")
        self.g("config", "commit.gpgsign", "false")
        self.g("config", "core.hooksPath", str(self.root / "no-hooks"))
        (self.repo / "README.md").write_text("SpecCraft\n", encoding="utf-8")
        (self.repo / ".gitignore").write_text("records/\n.env\n", encoding="utf-8")
        self.g("add", "--all")
        self.g("commit", "-m", "initial")
        self.g("remote", "add", "origin", MODULE.EXPECTED_REMOTE)
        self.g("push", str(self.remote), "main:main")

    def g(self, *args, **kwargs):
        return self.original_git(self.repo, *args, **kwargs)

    def transport(self, repo, *args, **kwargs):
        args = list(args)
        if args[0] == "fetch":
            args[args.index("origin")] = str(self.remote)
        elif "push" in args and "origin" in args:
            args[args.index("origin")] = str(self.remote)
            args.remove("--set-upstream")
        elif args[0] == "ls-remote":
            args[args.index("origin")] = str(self.remote)
        return self.original_git(repo, *args, **kwargs)

    def run_sync(self, dry=False):
        args = argparse.Namespace(repo=self.repo, message="test sync", dry_run=dry, yes=True)
        with patch.object(MODULE, "git", side_effect=self.transport), contextlib.redirect_stdout(io.StringIO()):
            MODULE.sync(args)

    def head(self):
        return self.g("rev-parse", "HEAD").stdout.strip()

    def test_preview_preserves_staging_and_commit(self):
        (self.repo / "new.py").write_text("print('hello')\n", encoding="utf-8")
        before_head = self.head()
        before_status = self.g("status", "--porcelain").stdout
        self.run_sync(dry=True)
        self.assertEqual(before_head, self.head())
        self.assertEqual(before_status, self.g("status", "--porcelain").stdout)

    def test_commit_push_and_noop_retry(self):
        (self.repo / "new.py").write_text("print('hello')\n", encoding="utf-8")
        self.run_sync()
        pushed = self.original_git(self.remote, "rev-parse", "main").stdout.strip()
        self.assertEqual(self.head(), pushed)
        head = self.head()
        self.run_sync()
        self.assertEqual(head, self.head())

    def test_ignored_data_stays_local(self):
        (self.repo / "records").mkdir()
        (self.repo / "records" / "secret.txt").write_text("local evidence", encoding="utf-8")
        (self.repo / "new.py").write_text("pass\n", encoding="utf-8")
        self.run_sync()
        self.assertNotIn(b"records/", self.g("ls-files").stdout)

    def test_sensitive_filename_blocked(self):
        (self.repo / "private.pem").write_text("test", encoding="utf-8")
        with self.assertRaisesRegex(MODULE.SyncError, "Upload blocked"):
            self.run_sync()
        self.assertEqual(b"", self.g("diff", "--cached", "--name-only").stdout)

    def test_token_content_not_printed(self):
        token = "ghp_" + "x" * 40
        (self.repo / "config.txt").write_text(token, encoding="utf-8")
        with self.assertRaises(MODULE.SyncError) as caught:
            self.run_sync(dry=True)
        self.assertNotIn(token, str(caught.exception))

    def test_large_file_blocked(self):
        (self.repo / "large.bin").write_bytes(b"a" * 64)
        with patch.object(MODULE, "MAX_BYTES", 32), self.assertRaisesRegex(MODULE.SyncError, "larger"):
            self.run_sync(dry=True)

    def test_wrong_remote_blocked(self):
        self.g("remote", "set-url", "origin", "https://github.com/example/wrong.git")
        with self.assertRaisesRegex(MODULE.SyncError, "Unexpected origin"):
            self.run_sync()

    def test_mirror_push_blocked(self):
        self.g("config", "remote.origin.mirror", "true")
        with self.assertRaisesRegex(MODULE.SyncError, "Mirror"):
            self.run_sync()

    def test_remote_ahead_preserves_work(self):
        old_head = self.head()
        (self.repo / "remote-change.txt").write_text("remote", encoding="utf-8")
        self.g("add", "--all")
        self.g("commit", "-m", "remote advances")
        self.g("push", str(self.remote), "main:main")
        # Soft reset only inside the disposable test repository; no files are deleted.
        self.g("reset", "--soft", old_head.decode())
        staged = self.g("diff", "--cached", "--name-only").stdout
        with self.assertRaisesRegex(MODULE.SyncError, "Remote is ahead"):
            self.run_sync()
        self.assertEqual(old_head, self.head())
        self.assertEqual(staged, self.g("diff", "--cached", "--name-only").stdout)

    def test_secret_in_earlier_outgoing_commit_blocked(self):
        before = self.original_git(self.remote, "rev-parse", "main").stdout
        (self.repo / "bad.txt").write_text("ghp_" + "x" * 40, encoding="utf-8")
        self.g("add", "--all")
        self.g("commit", "-m", "bad local commit")
        (self.repo / "bad.txt").unlink()
        self.g("add", "--all")
        self.g("commit", "-m", "delete secret")
        with self.assertRaisesRegex(MODULE.SyncError, "Upload blocked"):
            self.run_sync()
        self.assertEqual(before, self.original_git(self.remote, "rev-parse", "main").stdout)


if __name__ == "__main__":
    unittest.main()

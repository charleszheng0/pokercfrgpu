"""Temporary, isolated project snapshot and narrowly scoped agent tools."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


MAX_READ_BYTES = 160_000
MAX_PATCH_BYTES = 300_000
MAX_COMMAND_SECONDS = 240
IGNORED_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
}
BLOCKED_PARTS = {".git", ".ssh"}
BLOCKED_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


class IsolatedWorkspace:
    """A disposable copy of the repository; agent edits never touch the source tree."""

    def __init__(self, source_root: Path, keep: bool = False) -> None:
        self.source_root = source_root.resolve()
        if not (self.source_root / ".git").exists():
            raise ValueError(f"Not a Git repository: {self.source_root}")
        self._temporary = None if keep else tempfile.TemporaryDirectory(prefix="poker-agents-")
        self.root = (
            Path(tempfile.mkdtemp(prefix="poker-agents-"))
            if keep
            else Path(self._temporary.name)
        )
        self._snapshot_path = (self.root / "workspace").resolve()
        self._copy_snapshot()
        self._initialize_snapshot_git()

    def _copy_snapshot(self) -> None:
        def ignore(directory: str, names: list[str]) -> set[str]:
            ignored = set(names).intersection(IGNORED_DIRS)
            ignored.update(name for name in names if name == ".DS_Store")
            return ignored

        shutil.copytree(self.source_root, self._snapshot_path, ignore=ignore)
        original_modules = self.source_root / "node_modules"
        if original_modules.is_dir():
            # Browser dependencies are read from the existing install; source edits
            # remain confined to the copied workspace.
            (self._snapshot_path / "node_modules").symlink_to(original_modules, target_is_directory=True)

    def _initialize_snapshot_git(self) -> None:
        self._git(["init", "-q"])
        self._git(["config", "user.email", "agent-workflow@localhost"])
        self._git(["config", "user.name", "Poker Agent Workflow"])
        self._git(["add", "-A"])
        self._git(["commit", "-qm", "Snapshot before agent run"])

    def _git(self, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self._snapshot_path,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )

    def safe_path(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or not relative_path or ".." in candidate.parts:
            raise ValueError("Use a non-empty repository-relative path without '..'.")
        if any(part in BLOCKED_PARTS for part in candidate.parts):
            raise ValueError("Access to Git metadata and SSH files is blocked.")
        if candidate.name in BLOCKED_NAMES or candidate.suffix.lower() in BLOCKED_SUFFIXES:
            raise ValueError("Access to likely credential files is blocked.")
        resolved = (self._snapshot_path / candidate).resolve()
        if not resolved.is_relative_to(self._snapshot_path.resolve()):
            raise ValueError("Path escapes the isolated project snapshot.")
        return resolved

    @property
    def workspace_root(self) -> Path:
        return self._snapshot_path

    def status(self) -> str:
        result = self._git(["status", "--short"])
        return result.stdout.strip() or "(clean snapshot)"

    def list_files(self, subdirectory: str = ".") -> str:
        path = self.safe_path(subdirectory)
        if not path.exists():
            return f"Not found: {subdirectory}"
        files: list[str] = []
        for directory, dirnames, filenames in os.walk(path):
            dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_DIRS)
            for filename in sorted(filenames):
                full_path = Path(directory) / filename
                files.append(str(full_path.relative_to(self._snapshot_path)))
                if len(files) >= 400:
                    files.append("… (truncated at 400 files)")
                    return "\n".join(files)
        return "\n".join(files) or "(no files)"

    def read_file(self, relative_path: str) -> str:
        path = self.safe_path(relative_path)
        if not path.is_file():
            return f"Not a file: {relative_path}"
        data = path.read_bytes()
        if len(data) > MAX_READ_BYTES:
            return f"File is {len(data)} bytes; read limit is {MAX_READ_BYTES}. Request a smaller relevant file."
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return f"Binary file ({len(data)} bytes); text reading is unavailable."

    def apply_patch(self, patch: str) -> str:
        encoded = patch.encode("utf-8")
        if len(encoded) > MAX_PATCH_BYTES:
            return f"Patch exceeds the {MAX_PATCH_BYTES}-byte limit. Split it into smaller changes."
        if not patch.startswith("diff --git "):
            return "Rejected: provide a standard unified Git diff beginning with 'diff --git'."
        for line in patch.splitlines():
            if line.startswith(("--- ", "+++ ")):
                target = line[4:].split("\t", 1)[0]
                if target not in {"/dev/null"}:
                    target = target.removeprefix("a/").removeprefix("b/")
                    try:
                        self.safe_path(target)
                    except ValueError as exc:
                        return f"Rejected patch path {target!r}: {exc}"
        # git apply reads patch text from stdin, so run it directly here.
        checked = subprocess.run(
            ["git", "apply", "--check", "-"],
            cwd=self._snapshot_path,
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        if checked.returncode:
            return f"Patch check failed; nothing was changed.\n{checked.stdout[-6000:]}"
        applied = subprocess.run(
            ["git", "apply", "-"],
            cwd=self._snapshot_path,
            input=patch,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            check=False,
        )
        if applied.returncode:
            return f"Patch application failed.\n{applied.stdout[-6000:]}"
        return "Patch applied to the isolated snapshot only.\n" + self.status()

    def diff(self) -> str:
        # Stage only inside the disposable snapshot so newly created files appear
        # in the same review/export diff as edits to tracked files.
        added = self._git(["add", "-A"])
        if added.returncode:
            return f"Could not stage snapshot changes for diff:\n{added.stdout[-6000:]}"
        result = self._git(["diff", "--cached", "--no-ext-diff", "--binary", "HEAD", "--"])
        return result.stdout or "(no changes)"

    def run_check(self, check_id: str) -> str:
        commands: dict[str, list[str]] = {
            "python_tests": [sys.executable, "-m", "pytest", "-q"],
            "browser_demo": ["npm", "run", "test:browser"],
            "browser_showcase": ["npm", "run", "test:showcase"],
        }
        command = commands.get(check_id)
        if command is None:
            return "Unknown check. Allowed IDs: python_tests, browser_demo, browser_showcase."
        check_env = os.environ.copy()
        for name in tuple(check_env):
            upper = name.upper()
            if any(marker in upper for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")):
                check_env.pop(name, None)
        try:
            result = subprocess.run(
                command,
                cwd=self._snapshot_path,
                env=check_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=MAX_COMMAND_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            tail = (exc.stdout or "")
            if isinstance(tail, bytes):
                tail = tail.decode("utf-8", errors="replace")
            return f"{check_id}: timed out after {MAX_COMMAND_SECONDS}s.\n{tail[-6000:]}"
        except FileNotFoundError as exc:
            return f"{check_id}: command unavailable: {exc}"
        output = result.stdout[-12_000:]
        return f"{check_id}: exit={result.returncode}\n{output or '(no output)'}"

    def save_patch(self) -> Path | None:
        patch = self.diff()
        if patch == "(no changes)":
            return None
        out_dir = self.source_root / "artifacts" / "agent_runs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = out_dir / f"agent-run-{stamp}.patch"
        counter = 1
        while output.exists():
            output = out_dir / f"agent-run-{stamp}-{counter}.patch"
            counter += 1
        output.write_text(patch, encoding="utf-8")
        return output

    def close(self) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()

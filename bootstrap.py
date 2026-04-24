#!/usr/bin/env python3
"""
Shared bootstrap logic for local workshop setup on macOS, Linux, and Windows.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENDOR_DIR = ROOT / "vendor"
FW_DIR = VENDOR_DIR / "nflows_reflectorch"

FW_REPO_URL = "https://github.com/m-levytskyi/reflectorch-nflows"
FW_REF = "dev_ml"

REQ_FILE = ROOT / "requirements.txt"
TORCH_REQ_CU118 = ROOT / "requirements.torch-cu118.txt"
TORCH_REQ_CU121 = ROOT / "requirements.torch-cu121.txt"

MIN_PYTHON = (3, 10)
MAX_PYTHON = (3, 12)


class BootstrapError(RuntimeError):
    def __init__(
        self,
        step: str,
        message: str,
        *,
        command: Sequence[str] | None = None,
        cwd: Path | None = None,
        likely_cause: str | None = None,
        next_step: str | None = None,
    ) -> None:
        super().__init__(message)
        self.step = step
        self.message = message
        self.command = list(command) if command else None
        self.cwd = cwd
        self.likely_cause = likely_cause
        self.next_step = next_step

    def format(self) -> str:
        lines = [f"ERROR in step: {self.step}", self.message]
        if self.command:
            lines.append(f"Command: {' '.join(str(part) for part in self.command)}")
        if self.cwd:
            lines.append(f"Working directory: {self.cwd}")
        if self.likely_cause:
            lines.append(f"Likely cause: {self.likely_cause}")
        if self.next_step:
            lines.append(f"Next step: {self.next_step}")
        return "\n".join(lines)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def venv_pip() -> Path:
    return VENV / ("Scripts/pip.exe" if sys.platform == "win32" else "bin/pip")


def which(name: str) -> str | None:
    return shutil.which(name)


def uv_available() -> bool:
    return bool(which("uv"))


def announce(step_number: int, total_steps: int, label: str) -> None:
    print(f"[{step_number}/{total_steps}] {label}", flush=True)


def run(
    cmd: Sequence[str],
    *,
    step: str,
    cwd: Path | None = None,
    likely_cause: str | None = None,
    next_step: str | None = None,
) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    try:
        subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            check=True,
        )
    except FileNotFoundError as exc:
        raise BootstrapError(
            step,
            f"Required command was not found: {exc.filename}",
            command=cmd,
            cwd=cwd,
            likely_cause=likely_cause or "The required tool is not installed or not on PATH.",
            next_step=next_step or "Install the missing tool, open a new terminal, and retry.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(
            step,
            f"Command exited with status {exc.returncode}. See the command output above for details.",
            command=cmd,
            cwd=cwd,
            likely_cause=likely_cause,
            next_step=next_step,
        ) from exc


def capture(
    cmd: Sequence[str],
    *,
    step: str,
    cwd: Path | None = None,
    likely_cause: str | None = None,
    next_step: str | None = None,
) -> str:
    try:
        result = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise BootstrapError(
            step,
            f"Required command was not found: {exc.filename}",
            command=cmd,
            cwd=cwd,
            likely_cause=likely_cause or "The required tool is not installed or not on PATH.",
            next_step=next_step or "Install the missing tool, open a new terminal, and retry.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise BootstrapError(
            step,
            f"Command exited with status {exc.returncode}.",
            command=cmd,
            cwd=cwd,
            likely_cause=likely_cause,
            next_step=next_step,
        ) from exc
    return result.stdout.strip()


def require_python_version() -> None:
    current = sys.version_info[:3]
    if not (MIN_PYTHON <= sys.version_info[:2] <= MAX_PYTHON):
        raise BootstrapError(
            "Checking prerequisites",
            (
                f"Unsupported Python version detected: {current[0]}.{current[1]}.{current[2]}. "
                f"This project supports Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} to "
                f"{MAX_PYTHON[0]}.{MAX_PYTHON[1]}."
            ),
            likely_cause="The workshop setup was launched with a different Python version than the project supports.",
            next_step="Install Python 3.10, 3.11, or 3.12 and rerun the same command.",
        )


def require_command(name: str, *, pretty_name: str, install_hint: str) -> str:
    path = which(name)
    if path:
        return path

    raise BootstrapError(
        "Checking prerequisites",
        f"Missing required tool: {pretty_name}",
        likely_cause=f"{pretty_name} is not installed or is not available on PATH.",
        next_step=install_hint,
    )


def check_tools() -> None:
    require_python_version()
    git_path = require_command(
        "git",
        pretty_name="Git",
        install_hint="Install Git, then reopen the terminal so `git` is available on PATH.",
    )
    lfs_path = require_command(
        "git-lfs",
        pretty_name="Git LFS",
        install_hint=(
            "Install Git LFS, run `git lfs install` once, then reopen the terminal and retry."
        ),
    )
    installer = "uv" if uv_available() else "pip"

    print(f"OK: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} at {sys.executable}")
    print(f"OK: Git at {git_path}")
    print(f"OK: Git LFS at {lfs_path}")
    print(f"OK: Installer set to {installer}")


def check_lfs() -> None:
    require_python_version()
    lfs_path = require_command(
        "git-lfs",
        pretty_name="Git LFS",
        install_hint=(
            "Install Git LFS, run `git lfs install` once, then reopen the terminal and retry."
        ),
    )
    print(f"OK: Git LFS at {lfs_path}")


def create_venv() -> None:
    require_python_version()

    if VENV.exists() and not venv_python().exists():
        raise BootstrapError(
            "Creating virtual environment",
            f"{VENV} already exists but does not contain a valid Python executable.",
            likely_cause="A previous setup attempt created a partial or corrupted virtual environment.",
            next_step="Delete `.venv` and rerun `make setup` or `python bootstrap_windows.py`.",
        )

    if not VENV.exists():
        if uv_available():
            run(
                ["uv", "venv", "--python", sys.executable, str(VENV)],
                step="Creating virtual environment",
                likely_cause="`uv` could not create the virtual environment with the current Python interpreter.",
                next_step="Check that the current Python installation is healthy, or retry without `uv` installed.",
            )
        else:
            run(
                [sys.executable, "-m", "venv", str(VENV)],
                step="Creating virtual environment",
                likely_cause="The standard library `venv` module failed while creating `.venv`.",
                next_step="Ensure your Python installation includes `venv`, then rerun the command.",
            )

    ensure_base_packages()


def ensure_venv_exists() -> None:
    if not venv_python().exists():
        raise BootstrapError(
            "Using virtual environment",
            "The local virtual environment does not exist yet.",
            likely_cause="Setup has not been run yet, or `.venv` was removed.",
            next_step="Run `make venv` or `make setup` first.",
        )


def pip_install(args: Sequence[str], *, step: str, likely_cause: str, next_step: str) -> None:
    ensure_venv_exists()
    if uv_available():
        run(
            ["uv", "pip", "install", "--python", str(venv_python()), *args],
            step=step,
            likely_cause=likely_cause,
            next_step=next_step,
        )
    else:
        run(
            [str(venv_pip()), "install", *args],
            step=step,
            likely_cause=likely_cause,
            next_step=next_step,
        )


def ensure_base_packages() -> None:
    pip_install(
        ["-U", "pip", "wheel", "setuptools"],
        step="Upgrading packaging tools",
        likely_cause="The package installer could not update core packaging tools inside `.venv`.",
        next_step="Check your network connection and Python packaging configuration, then retry.",
    )


def ensure_framework_repo_usable() -> None:
    if FW_DIR.exists() and not (FW_DIR / ".git").exists():
        raise BootstrapError(
            "Preparing framework checkout",
            f"{FW_DIR} already exists but is not a Git repository.",
            likely_cause="That directory was created manually or left behind by an incomplete clone.",
            next_step=f"Remove or rename `{FW_DIR}` and rerun setup.",
        )


def ensure_framework_repo_exists() -> None:
    if not (FW_DIR / ".git").exists():
        raise BootstrapError(
            "Using framework checkout",
            f"The managed framework repository is missing at {FW_DIR}.",
            likely_cause="The framework clone step has not completed yet.",
            next_step="Run `make framework` or `make setup` first.",
        )


def ensure_framework_clean() -> None:
    ensure_framework_repo_exists()
    status = capture(
        ["git", "status", "--short"],
        step="Checking framework repository state",
        cwd=FW_DIR,
        likely_cause="Git could not inspect the existing framework checkout.",
        next_step="Verify the checkout is intact, or delete `vendor/nflows_reflectorch` and rerun setup.",
    )
    if status:
        preview = "\n".join(status.splitlines()[:10])
        raise BootstrapError(
            "Checking framework repository state",
            (
                "The managed framework checkout has local changes, so setup will not overwrite it.\n"
                f"Detected changes:\n{preview}"
            ),
            likely_cause="Someone modified files inside `vendor/nflows_reflectorch` after cloning.",
            next_step=(
                "Commit or discard those changes in `vendor/nflows_reflectorch`, or remove that directory and rerun setup."
            ),
        )


def clone_or_update_framework() -> None:
    require_python_version()
    require_command(
        "git",
        pretty_name="Git",
        install_hint="Install Git, then reopen the terminal so `git` is available on PATH.",
    )
    ensure_framework_repo_usable()
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    if not (FW_DIR / ".git").exists():
        run(
            ["git", "clone", FW_REPO_URL, str(FW_DIR)],
            step="Cloning framework repository",
            likely_cause="Git could not clone the pinned framework repository.",
            next_step="Check your network connection and repository access, then retry.",
        )
    else:
        ensure_framework_clean()

    run(
        ["git", "fetch", "--all", "--tags", "--prune"],
        cwd=FW_DIR,
        step="Updating framework repository",
        likely_cause="Git could not fetch the latest refs for the managed framework checkout.",
        next_step="Check your network connection and try again.",
    )
    run(
        ["git", "checkout", FW_REF],
        cwd=FW_DIR,
        step="Checking out pinned framework ref",
        likely_cause=f"The pinned framework ref `{FW_REF}` could not be checked out.",
        next_step="Verify that the repository still contains that ref, or update the pinned ref in `bootstrap.py`.",
    )
    run(
        ["git", "pull", "--ff-only", "origin", FW_REF],
        cwd=FW_DIR,
        step="Fast-forwarding framework repository",
        likely_cause="The local framework branch cannot be fast-forwarded cleanly to the pinned remote branch.",
        next_step=(
            "Inspect `vendor/nflows_reflectorch`, resolve any local branch state issues, or delete the directory and rerun setup."
        ),
    )


def pull_lfs() -> None:
    require_python_version()
    require_command(
        "git-lfs",
        pretty_name="Git LFS",
        install_hint="Install Git LFS, run `git lfs install` once, then reopen the terminal and retry.",
    )
    ensure_framework_repo_exists()
    run(
        ["git", "lfs", "pull"],
        cwd=FW_DIR,
        step="Pulling Git LFS assets",
        likely_cause=(
            "Git LFS is missing, not initialized, or the network failed while downloading the framework model files."
        ),
        next_step="Run `git lfs install`, verify network access, and rerun setup.",
    )


def install_framework() -> None:
    require_python_version()
    ensure_venv_exists()
    ensure_framework_repo_exists()
    pip_install(
        ["-e", str(FW_DIR)],
        step="Installing framework package",
        likely_cause="The editable install of `vendor/nflows_reflectorch` failed.",
        next_step="Inspect the command output above for packaging errors, then retry after fixing them.",
    )


def install_torch(torch_wheel: str) -> None:
    require_python_version()
    ensure_venv_exists()

    if torch_wheel == "cpu":
        pip_install(
            [
                "--upgrade",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            step="Installing PyTorch",
            likely_cause="PyTorch CPU wheels could not be downloaded or installed.",
            next_step="Check your network connection, then retry. If you need a CUDA build, rerun with `TORCH_WHEEL=cu121` or `--torch-wheel cu121`.",
        )
        return

    if torch_wheel == "cu118":
        req_file = TORCH_REQ_CU118
    elif torch_wheel == "cu121":
        req_file = TORCH_REQ_CU121
    else:
        raise BootstrapError(
            "Installing PyTorch",
            f"Unsupported torch wheel selection: {torch_wheel}",
            likely_cause="The bootstrap command received an unsupported torch wheel value.",
            next_step="Use one of: cpu, cu118, cu121.",
        )

    pip_install(
        ["-r", str(req_file)],
        step="Installing PyTorch",
        likely_cause=f"The PyTorch requirements file `{req_file.name}` could not be installed.",
        next_step="Verify that your machine supports that CUDA variant, or retry with the default CPU wheel.",
    )


def install_deps() -> None:
    require_python_version()
    ensure_venv_exists()

    if not REQ_FILE.exists():
        print("No requirements.txt found; skipping base dependency installation.")
        return

    pip_install(
        ["-r", str(REQ_FILE)],
        step="Installing project dependencies",
        likely_cause="The project requirements could not be installed into `.venv`.",
        next_step="Check the package installation error above, then retry after fixing the dependency issue.",
    )


def check_torch() -> None:
    require_python_version()
    ensure_venv_exists()
    code = (
        "import torch; "
        "print('torch', torch.__version__); "
        "print('cuda available', torch.cuda.is_available()); "
        "print('mps available', getattr(getattr(torch.backends, 'mps', None), 'is_available', lambda: False)()); "
        "print('device', 'cuda' if torch.cuda.is_available() else "
        "('mps' if getattr(getattr(torch.backends, 'mps', None), 'is_available', lambda: False)() else 'cpu'))"
    )
    run(
        [str(venv_python()), "-c", code],
        step="Checking PyTorch backend",
        likely_cause="PyTorch is not installed correctly inside `.venv`, or the selected backend is unavailable.",
        next_step="Rerun `make torch` with a different wheel if needed, then retry `make check-torch`.",
    )


def clean() -> None:
    if VENV.exists():
        shutil.rmtree(VENV)
        print(f"Removed {VENV}")
    else:
        print(f"Nothing to remove: {VENV}")


def distclean() -> None:
    clean()
    if VENDOR_DIR.exists():
        shutil.rmtree(VENDOR_DIR)
        print(f"Removed {VENDOR_DIR}")
    else:
        print(f"Nothing to remove: {VENDOR_DIR}")


def run_setup(torch_wheel: str) -> None:
    steps = [
        ("Checking prerequisites", check_tools),
        ("Creating virtual environment", create_venv),
        ("Cloning or updating framework", clone_or_update_framework),
        ("Pulling Git LFS assets", pull_lfs),
        ("Installing framework package", install_framework),
        ("Installing PyTorch", lambda: install_torch(torch_wheel)),
        ("Installing project dependencies", install_deps),
        ("Checking PyTorch backend", check_torch),
    ]

    for index, (label, action) in enumerate(steps, start=1):
        announce(index, len(steps), label)
        action()


def run_single_action(action: str, torch_wheel: str) -> None:
    actions = {
        "check-tools": ("Checking prerequisites", check_tools),
        "check-lfs": ("Checking Git LFS", check_lfs),
        "venv": ("Creating virtual environment", create_venv),
        "framework": ("Cloning or updating framework", clone_or_update_framework),
        "lfs": ("Pulling Git LFS assets", pull_lfs),
        "install-framework": ("Installing framework package", install_framework),
        "torch": ("Installing PyTorch", lambda: install_torch(torch_wheel)),
        "deps": ("Installing project dependencies", install_deps),
        "check-torch": ("Checking PyTorch backend", check_torch),
        "clean": ("Removing virtual environment", clean),
        "distclean": ("Removing virtual environment and vendor checkout", distclean),
    }
    label, callback = actions[action]
    announce(1, 1, label)
    callback()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared bootstrap helper for repository setup and maintenance targets."
    )
    parser.add_argument(
        "action",
        choices=[
            "setup",
            "check-tools",
            "check-lfs",
            "venv",
            "framework",
            "lfs",
            "install-framework",
            "torch",
            "deps",
            "check-torch",
            "clean",
            "distclean",
        ],
        help="Bootstrap action to execute.",
    )
    parser.add_argument(
        "--torch-wheel",
        default="cpu",
        choices=["cpu", "cu118", "cu121"],
        help="PyTorch wheel variant to install when the selected action needs it.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.action == "setup":
            run_setup(args.torch_wheel)
        else:
            run_single_action(args.action, args.torch_wheel)
    except BootstrapError as exc:
        print(exc.format(), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

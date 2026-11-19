import shutil
import subprocess
from unittest.mock import patch

import pytest

import bootstrap


@pytest.mark.parametrize(
    "cuda,expected",
    [
        ("11.7", "cpu"),
        ("11.8", "cu118"),
        ("12.1", "cu118"),
        ("12.5", "cu118"),
        ("12.6", "cu126"),
        ("12.8", "cu128"),
    ],
)
def test_cuda_selection_only_uses_available_pinned_builds(cuda, expected):
    selected = bootstrap.resolve_torch_backend(
        "auto",
        platform_name="linux",
        machine="x86_64",
        nvidia_smi_output=f"CUDA Version: {cuda}",
    )
    assert selected.resolved == expected


def test_framework_checkout_is_pinned_and_never_pulls_a_branch(tmp_path):
    framework = tmp_path / "framework"
    (framework / ".git").mkdir(parents=True)
    with (
        patch.object(bootstrap, "FW_DIR", framework),
        patch.object(bootstrap, "VENDOR_DIR", tmp_path),
        patch.object(bootstrap, "require_python_version"),
        patch.object(bootstrap, "require_command"),
        patch.object(bootstrap, "ensure_framework_clean"),
        patch.object(bootstrap, "run") as run,
    ):
        bootstrap.clone_or_update_framework()
    commands = [call.args[0] for call in run.call_args_list]
    assert ["git", "checkout", "--detach", bootstrap.FW_REF] in commands
    assert len(bootstrap.FW_REF) == 40
    assert all(command[1] != "pull" for command in commands)


@pytest.mark.skipif(shutil.which("make") is None, reason="Make is a Unix workflow")
@pytest.mark.parametrize(
    "target,action",
    [
        ("test", "test"),
        ("lint", "lint"),
        ("type-check", "type-check"),
        ("pre-commit", "pre-commit"),
    ],
)
def test_quality_targets_do_not_install_dependencies(target, action):
    result = subprocess.run(
        ["make", "-n", target],
        cwd=bootstrap.ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"bootstrap.py {action}" in result.stdout
    for install_action in ("venv", "torch", "deps", "dev-deps"):
        assert f"bootstrap.py {install_action}" not in result.stdout


def test_quality_command_propagates_failure():
    with (
        patch.object(bootstrap, "require_python_version"),
        patch.object(bootstrap, "ensure_venv_exists"),
        patch.object(
            bootstrap, "run", side_effect=bootstrap.BootstrapError("tests", "failed")
        ),
    ):
        with pytest.raises(bootstrap.BootstrapError, match="failed"):
            bootstrap.run_tests()

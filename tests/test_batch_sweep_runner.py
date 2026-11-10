import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

import batch_sweep_runner as sweep


def config(tmp_path, **overrides):
    result = {
        "sweep_results_dir": str(tmp_path / "sweeps"),
        "prior_deviations": [30],
        "sld_modes": ["none"],
        "prominent_features": [False],
    }
    result.update(overrides)
    return result


def test_forwards_options_and_links_actual_child_directory(tmp_path, monkeypatch):
    def child(cmd, **kwargs):
        output_root = Path(cmd[cmd.index("--output-dir") + 1])
        (output_root / "001_batch").mkdir()
        (output_root / "001_batch/run_metadata.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    launch = Mock(side_effect=child)
    monkeypatch.setattr(sweep.subprocess, "run", launch)
    runner = sweep.BatchPipelineSweep(
        config(
            tmp_path,
            disable_preprocessing=True,
            disable_constraints=True,
            experiment_ids=["s002", "s001"],
            num_experiments=1,
            use_sigmas_as_input=True,
            use_theoretical=True,
        )
    )
    result = runner.run_single_batch(30, "none", True)
    cmd = launch.call_args.args[0]
    assert "--disable-preprocessing" in cmd
    assert "--disable-constraints" in cmd
    assert "--use-prominent-features" in cmd
    assert "--use-sigmas-input" in cmd
    assert "--use-theoretical" in cmd
    offset = cmd.index("--experiment-ids")
    assert cmd[offset + 1 : offset + 3] == ["s002", "s001"]
    assert cmd[cmd.index("--num-experiments") + 1] == "1"
    assert result["output_dir"].endswith("run_001/001_batch")
    assert Path(result["output_dir"]).is_dir()
    saved = json.loads((runner.sweep_dir / "sweep_summary.json").read_text())
    assert saved["results"][0]["output_dir"] == result["output_dir"]


def test_predict_model_override_reaches_child(tmp_path, monkeypatch):
    launch = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(sweep.subprocess, "run", launch)
    runner = sweep.BatchPipelineSweep(
        config(tmp_path, inference_backend="predict", config_name="custom_point.yaml")
    )
    runner.run_single_batch(30, "none", False)
    cmd = launch.call_args.args[0]
    assert cmd[cmd.index("--config-name") + 1] == "custom_point.yaml"
    assert "--nf-num-samples" not in cmd


def test_failed_run_does_not_stop_remaining_sweep(tmp_path, monkeypatch):
    launch = Mock(
        side_effect=[
            subprocess.CalledProcessError(1, ["batch"], stderr="bad data"),
            SimpleNamespace(returncode=0),
        ]
    )
    monkeypatch.setattr(sweep.subprocess, "run", launch)
    runner = sweep.BatchPipelineSweep(config(tmp_path, prior_deviations=[5, 30]))
    results = runner.run_full_sweep()
    assert [result["success"] for result in results] == [False, True]
    assert launch.call_count == 2
    assert results[0]["error_output"] == "bad data"
    assert results[0]["output_dir"] != results[1]["output_dir"]


def test_timeout_keeps_child_records_and_output_link(tmp_path, monkeypatch):
    def child(cmd, **kwargs):
        output_root = Path(cmd[cmd.index("--output-dir") + 1])
        records = output_root / "001_partial/experiments"
        records.mkdir(parents=True)
        (records / "s001.json").write_text('{"success": true}')
        raise subprocess.TimeoutExpired(cmd, 7200)

    monkeypatch.setattr(sweep.subprocess, "run", child)
    runner = sweep.BatchPipelineSweep(config(tmp_path))
    result = runner.run_single_batch(30, "none", False)
    assert not result["success"]
    assert result["return_code"] == "TIMEOUT"
    assert json.loads(
        (Path(result["output_dir"]) / "experiments/s001.json").read_text()
    )["success"]


def test_cli_yaml_overrides_and_failed_sweep_exit_code(tmp_path, monkeypatch):
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(yaml.safe_dump(config(tmp_path, num_experiments=20)))
    launch = Mock(
        side_effect=subprocess.CalledProcessError(1, ["batch"], stderr="bad data")
    )
    monkeypatch.setattr(sweep.subprocess, "run", launch)
    monkeypatch.setattr(
        sweep.sys,
        "argv",
        [
            "batch_sweep_runner.py",
            "--config",
            str(config_file),
            "--num-experiments",
            "0",
            "--disable-preprocessing",
            "--disable-constraints",
            "--experiment-ids",
            "s001",
            "--config-name",
            "custom.yaml",
        ],
    )
    assert sweep.main() == 1
    cmd = launch.call_args.args[0]
    assert cmd[cmd.index("--num-experiments") + 1] == "0"
    assert "--disable-preprocessing" in cmd and "--disable-constraints" in cmd
    assert cmd[cmd.index("--experiment-ids") + 1] == "s001"
    assert cmd[cmd.index("--config-name") + 1] == "custom.yaml"


def test_interruption_records_current_run_and_returns_130(tmp_path, monkeypatch):
    config_file = tmp_path / "sweep.yaml"
    config_file.write_text(yaml.safe_dump(config(tmp_path, prior_deviations=[5, 30])))
    monkeypatch.setattr(
        sweep.sys, "argv", ["batch_sweep_runner.py", "--config", str(config_file)]
    )
    launch = Mock(side_effect=KeyboardInterrupt())
    monkeypatch.setattr(sweep.subprocess, "run", launch)
    assert sweep.main() == 130
    assert launch.call_count == 1
    summary_path = next((tmp_path / "sweeps").glob("*/sweep_summary.json"))
    result = json.loads(summary_path.read_text())["results"][0]
    assert result["return_code"] == 130
    assert not result["success"]
    assert Path(result["output_dir"]).exists()


@pytest.mark.parametrize(
    "invalid",
    [
        {"prior_deviations": []},
        {"prior_deviations": [42]},
        {"sld_modes": ["invalid"]},
        {"prominent_features": ["false"]},
    ],
)
def test_invalid_sweep_combinations_fail_before_launch(tmp_path, invalid):
    with pytest.raises(ValueError):
        sweep.BatchPipelineSweep(config(tmp_path, **invalid))


@pytest.mark.parametrize(
    "contents",
    ["", "# comment only\n", "null", "false", "[]", "[1, 2]", "not-a-mapping"],
)
def test_nonmapping_yaml_does_not_launch_default_sweep(tmp_path, monkeypatch, contents):
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text(contents)
    launch = Mock()
    monkeypatch.setattr(sweep.subprocess, "run", launch)
    monkeypatch.setattr(
        sweep.sys, "argv", ["batch_sweep_runner.py", "--config", str(config_file)]
    )
    assert sweep.main() == 1
    assert not launch.called
    assert list(tmp_path.iterdir()) == [config_file]


def test_explicit_empty_mapping_is_valid_config(tmp_path):
    config_file = tmp_path / "defaults.yaml"
    config_file.write_text("{}")
    assert sweep.load_config_from_yaml(config_file) == {}

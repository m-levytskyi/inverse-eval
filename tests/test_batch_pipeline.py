import hashlib
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import batch_pipeline as batch
from utils import write_json_atomic


def success(value=10):
    return {"param_metrics": {"overall": {"constraint_mape": value}}}


@pytest.fixture
def pipeline_factory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        batch, "_collect_provenance", lambda: {"repository": {"commit": "test"}}
    )
    monkeypatch.setattr(batch, "create_batch_analysis_plots", Mock())
    discovery = Mock(return_value=["s002", "s001"])
    monkeypatch.setattr(batch, "discover_batch_experiments", discovery)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "test.yaml").write_text("model config")
    (model_dir / "test.safetensors").write_bytes(b"weights")
    model = SimpleNamespace(
        device="cpu",
        config_dir=model_dir,
        config_name="test.yaml",
        model_dir=model_dir,
        model_name="test.safetensors",
    )
    model_loader = Mock(return_value=model)
    inference = Mock(side_effect=lambda **kwargs: success())
    monkeypatch.setattr(batch, "create_inference_model", model_loader)
    monkeypatch.setattr(batch, "run_single_experiment", inference)

    def make(**kwargs):
        options = {
            "output_dir": tmp_path / "results",
            "priors_type": "narrow",
            "config_name": "test.yaml",
        }
        options.update(kwargs)
        return batch.BatchInferencePipeline(**options)

    return SimpleNamespace(
        make=make, discovery=discovery, model_loader=model_loader, inference=inference
    )


def test_selection_deduplicates_and_applies_limit_after_prominent_filter(
    pipeline_factory, monkeypatch
):
    pipeline_factory.discovery.return_value = ["s003", "s001", "s002", "s002"]
    monkeypatch.setattr(
        batch,
        "find_experiments_with_prominent_peaks",
        Mock(return_value=["s003", "s002"]),
    )
    pipeline = pipeline_factory.make(num_experiments=1, use_prominent_features=True)
    assert pipeline.discover_experiments() == ["s002"]
    assert pipeline.num_experiments == 1
    assert pipeline_factory.discovery.call_args.kwargs["num_experiments"] is None


def test_explicit_ids_keep_requested_order_without_duplicates(pipeline_factory):
    pipeline_factory.discovery.return_value = ["s002", "s001", "s002"]
    pipeline = pipeline_factory.make(experiment_ids=["s002", "s001", "s002"])
    assert pipeline.discover_experiments() == ["s002", "s001"]
    assert pipeline_factory.discovery.call_args.kwargs["experiment_ids"] == [
        "s002",
        "s001",
    ]


def test_model_reused_and_all_artifacts_preserve_current_result_map(pipeline_factory):
    pipeline = pipeline_factory.make(inference_backend="nf", nf_num_samples=7)
    results = pipeline.run()
    assert list(results) == ["s001", "s002"]
    assert pipeline_factory.model_loader.call_count == 1
    model = pipeline_factory.model_loader.return_value
    assert all(
        call.kwargs["inference_model"] is model
        for call in pipeline_factory.inference.call_args_list
    )
    assert (
        json.loads((pipeline.output_dir / "batch_results.json").read_text()) == results
    )
    for experiment_id, result in results.items():
        assert (
            json.loads(
                (
                    pipeline.output_dir / "experiments" / f"{experiment_id}.json"
                ).read_text()
            )
            == result
        )
    metadata = json.loads((pipeline.output_dir / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["counts"] == {
        "completed": 2,
        "successful": 2,
        "failed": 0,
        "outliers": 0,
    }
    assert metadata["config"]["nf_num_samples"] == 7
    assert metadata["config"]["config_name"] == "test.yaml"
    assert metadata["config"]["inference_device"] == "cpu"
    assert (
        metadata["config"]["metric_policy"]
        == "constraint_mape_global_max_log_likelihood"
    )
    assert (
        metadata["provenance"]["model_weights"]["sha256"]
        == hashlib.sha256(b"weights").hexdigest()
    )


def test_failures_are_saved_alongside_successes(pipeline_factory):
    pipeline_factory.inference.side_effect = [success(), ValueError("bad data")]
    pipeline = pipeline_factory.make()
    results = pipeline.run()
    assert results["s001"]["success"]
    assert not results["s002"]["success"]
    assert (
        json.loads((pipeline.output_dir / "experiments/s002.json").read_text())["error"]
        == "bad data"
    )
    assert pipeline.run_metadata["status"] == "failed"
    summary = json.loads(
        (pipeline.output_dir / "batch_summary_1layer.json").read_text()
    )
    assert summary["total_experiments"] == 2
    assert summary["successful_experiments"] == 1
    assert summary["failed_experiments"] == 1


def test_failed_model_is_loaded_only_once(pipeline_factory):
    pipeline_factory.model_loader.side_effect = RuntimeError("missing weights")
    pipeline = pipeline_factory.make()
    results = pipeline.run()
    assert pipeline_factory.model_loader.call_count == 1
    assert not pipeline_factory.inference.called
    assert all("missing weights" in result["error"] for result in results.values())
    assert pipeline.run_metadata["counts"]["failed"] == 2


def test_interruption_saves_completed_records_and_partial_aggregate(pipeline_factory):
    pipeline_factory.inference.side_effect = [success(), KeyboardInterrupt()]
    pipeline = pipeline_factory.make()
    with pytest.raises(KeyboardInterrupt):
        pipeline.run()
    assert (pipeline.output_dir / "experiments/s001.json").exists()
    assert not (pipeline.output_dir / "experiments/s002.json").exists()
    assert list(
        json.loads((pipeline.output_dir / "batch_results.json").read_text())
    ) == ["s001"]
    assert pipeline.run_metadata["status"] == "interrupted"
    assert pipeline.run_metadata["counts"]["completed"] == 1


def test_all_outliers_complete_without_loading_model(pipeline_factory, monkeypatch):
    monkeypatch.setattr(
        batch,
        "discover_experiment_files",
        lambda *args, **kwargs: ("curve.dat", "model.dat", 1),
    )
    monkeypatch.setattr(batch, "parse_true_parameters_from_model_file", lambda path: {})
    monkeypatch.setattr(
        batch,
        "check_experiment_within_constraints",
        lambda *args: (False, [("thickness", 2000, 1, 1000)]),
    )
    pipeline = pipeline_factory.make(
        priors_type="constraint_based", narrow_priors_deviation=0.3
    )
    results = pipeline.run()
    assert all(result["excluded_as_outlier"] for result in results.values())
    assert not pipeline_factory.model_loader.called
    assert pipeline.run_metadata["status"] == "completed_no_eligible_results"
    assert pipeline.run_metadata["counts"]["outliers"] == 2
    assert len(list((pipeline.output_dir / "experiments").glob("*.json"))) == 2
    summary = json.loads(
        (pipeline.output_dir / "batch_summary_1layer.json").read_text()
    )
    assert summary["priors_deviation"] == 0.3
    assert "constraint_accuracy" not in summary


def test_no_experiments_is_a_failure_without_model_load(pipeline_factory):
    pipeline_factory.discovery.return_value = []
    pipeline = pipeline_factory.make()
    assert pipeline.run() is None
    assert pipeline.run_metadata["status"] == "failed"
    assert not pipeline_factory.model_loader.called


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_truth_is_saved_as_input_failure(
    pipeline_factory, monkeypatch, value
):
    monkeypatch.setattr(
        batch,
        "discover_experiment_files",
        lambda *args, **kwargs: ("curve.dat", "model.dat", 1),
    )
    monkeypatch.setattr(batch, "parse_true_parameters_from_model_file", lambda path: {})
    monkeypatch.setattr(
        batch,
        "check_experiment_within_constraints",
        lambda *args: (False, [("thickness", value, 1, 1000)]),
    )
    pipeline = pipeline_factory.make(priors_type="constraint_based")
    results = pipeline.run()
    assert all(result["error_type"] == "ValueError" for result in results.values())
    assert pipeline.run_metadata["status"] == "failed"
    assert pipeline.run_metadata["counts"]["failed"] == 2
    assert pipeline.run_metadata["counts"]["outliers"] == 0
    assert not pipeline_factory.model_loader.called
    for path in (pipeline.output_dir / "experiments").glob("*.json"):
        assert not json.loads(path.read_text())["success"]


def test_outlier_checkpoint_failure_is_not_swallowed(pipeline_factory, monkeypatch):
    monkeypatch.setattr(
        batch,
        "discover_experiment_files",
        lambda *args, **kwargs: ("curve.dat", "model.dat", 1),
    )
    monkeypatch.setattr(batch, "parse_true_parameters_from_model_file", lambda path: {})
    monkeypatch.setattr(
        batch,
        "check_experiment_within_constraints",
        lambda *args: (False, [("thickness", 2000, 1, 1000)]),
    )
    pipeline = pipeline_factory.make(priors_type="constraint_based")
    monkeypatch.setattr(
        pipeline, "_record_result", Mock(side_effect=OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        pipeline.run()
    assert pipeline.run_metadata["status"] == "failed"
    assert not pipeline_factory.model_loader.called


@pytest.mark.parametrize(
    "results, expected",
    [
        ({"a": {"success": True}}, 0),
        ({"a": {"success": False}}, 1),
        ({"a": {"success": True}, "b": {"success": False}}, 1),
        ({"a": {"success": False, "excluded_as_outlier": True}}, 0),
        (None, 1),
    ],
)
def test_cli_exit_status_reflects_processing_outcome(monkeypatch, results, expected):
    monkeypatch.setattr(batch.sys, "argv", ["batch_pipeline.py"])
    monkeypatch.setattr(
        batch,
        "BatchInferencePipeline",
        Mock(return_value=SimpleNamespace(run=lambda: results)),
    )
    assert batch.main() == expected


def test_cli_interruption_returns_130(monkeypatch):
    monkeypatch.setattr(batch.sys, "argv", ["batch_pipeline.py"])
    monkeypatch.setattr(
        batch,
        "BatchInferencePipeline",
        Mock(return_value=SimpleNamespace(run=Mock(side_effect=KeyboardInterrupt))),
    )
    assert batch.main() == 130


def test_result_ids_cannot_escape_output_directory(pipeline_factory):
    pipeline = pipeline_factory.make()
    with pytest.raises(ValueError, match="Invalid experiment ID"):
        pipeline._record_result("../escape", {"success": True})
    assert not (pipeline.output_dir / "escape.json").exists()


def test_atomic_write_preserves_previous_file_on_invalid_data_or_replace_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "record.json"
    write_json_atomic(path, {"original": True})
    with pytest.raises(ValueError):
        write_json_atomic(path, {"value": float("nan")})
    with monkeypatch.context() as context:
        context.setattr(os, "replace", Mock(side_effect=OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            write_json_atomic(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}
    assert list(tmp_path.iterdir()) == [path]

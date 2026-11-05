import pytest

from batch_analysis import (
    create_summary_statistics,
    print_constraint_mape_summary,
    print_summary_statistics,
)


def test_summary_counts_all_outcomes_and_only_scores_successes():
    records = {
        "ok": {"success": True, "param_metrics": {"overall": {"constraint_mape": 10}}},
        "bad": {
            "success": False,
            "param_metrics": {"overall": {"constraint_mape": -1}},
        },
        "excluded": {"success": False, "excluded_as_outlier": True},
    }
    summary = create_summary_statistics(
        records, 1, priors_type="constraint_based", narrow_priors_deviation=0.3
    )
    assert summary["total_experiments"] == 3
    assert summary["successful_experiments"] == 1
    assert summary["failed_experiments"] == 1
    assert summary["excluded_experiments"] == 1
    assert summary["priors_deviation"] == 0.3
    assert summary["constraint_accuracy"]["constraint_mape"]["mean"] == 10


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_success_metrics_fail_visibly(value):
    records = {
        "a": {"success": True, "param_metrics": {"overall": {"constraint_mape": value}}}
    }
    with pytest.raises(ValueError, match="finite, nonnegative"):
        create_summary_statistics(records, 1)
    with pytest.raises(ValueError, match="finite, nonnegative"):
        print_constraint_mape_summary(records)


def test_empty_summary_has_explicit_zero_counts():
    summary = create_summary_statistics({}, 1)
    assert summary["total_experiments"] == summary["successful_experiments"] == 0
    assert summary["failed_experiments"] == summary["excluded_experiments"] == 0
    assert "constraint_accuracy" not in summary


@pytest.mark.parametrize(
    "priors_type, label",
    [
        ("constraint_based", "Requested prior width: 30.0% of fixed evaluation range"),
        ("narrow", "Prior bounds deviation: +/-30.0%"),
    ],
)
def test_prior_width_summary_label_matches_prior_type(capsys, priors_type, label):
    summary = create_summary_statistics(
        {}, 1, priors_type=priors_type, narrow_priors_deviation=0.3
    )
    print_summary_statistics(summary)
    assert label in capsys.readouterr().out

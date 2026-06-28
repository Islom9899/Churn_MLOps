"""Tests for the MLflow production promotion gate."""

from src.pipeline.promote_model import metrics_pass


def test_metrics_pass_when_thresholds_are_met():
    """A model can be promoted when both required metrics pass."""
    assert metrics_pass({"roc_auc": 0.80, "f1": 0.55}, min_roc_auc=0.78, min_f1=0.50)


def test_metrics_fail_when_any_threshold_is_missed():
    """A model is blocked when any required metric is below threshold."""
    assert not metrics_pass({"roc_auc": 0.80, "f1": 0.49}, min_roc_auc=0.78, min_f1=0.50)
    assert not metrics_pass({"roc_auc": 0.77, "f1": 0.55}, min_roc_auc=0.78, min_f1=0.50)

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from plaintext_sims.metrics import (
    bootstrap_difference,
    format_summary_text,
    summarize_metrics,
)


def test_summarize_metrics_returns_expected_rows() -> None:
    df = pl.DataFrame(
        {
            "condition": ["plaintext", "mixed", "plaintext", "mixed"],
            "metric_one": [1.0, 2.0, 3.0, 4.0],
            "metric_two": [5.0, 6.0, 7.0, 8.0],
        }
    )
    summary = summarize_metrics(df, metrics=["metric_one", "metric_two"])

    assert set(summary.get_column("metric")) == {"metric_one", "metric_two"}
    assert summary.height == 4  # two metrics x two conditions
    metric_one_means = summary.filter(pl.col("metric") == "metric_one").get_column(
        "mean"
    )
    assert pytest.approx(metric_one_means.mean()) == 2.5


def test_bootstrap_difference_constant_gap() -> None:
    df = pl.DataFrame(
        {
            "condition": ["a"] * 20 + ["b"] * 20,
            "score": [1.0] * 20 + [2.0] * 20,
        }
    )
    rng = np.random.default_rng(0)

    result = bootstrap_difference(
        df,
        metric="score",
        condition_a="a",
        condition_b="b",
        reps=200,
        rng=rng,
    )

    assert set(result) == {"metric", "delta_mean", "ci_lower", "ci_upper"}
    assert result["metric"] == "score"
    assert result["delta_mean"] == pytest.approx(-1.0)
    assert result["ci_lower"] == pytest.approx(-1.0)
    assert result["ci_upper"] == pytest.approx(-1.0)


def test_format_summary_text_includes_metrics() -> None:
    df = pl.DataFrame(
        {
            "condition": ["plaintext", "mixed", "plaintext", "mixed"],
            "total_calendar_time": [10.0, 8.0, 10.0, 8.0],
            "num_rework_cycles": [1.0, 2.0, 1.0, 2.0],
            "num_late_defects": [0.5, 1.0, 0.5, 1.0],
            "handover_delay": [0.2, 0.3, 0.2, 0.3],
        }
    )

    text = format_summary_text(df)

    assert "total calendar time" in text
    assert "plaintext mean=10.0" in text
    assert "mixed mean=8.0" in text
    assert "mixed-plaintext=-2.0" in text

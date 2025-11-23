from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plaintext_sims import WorkflowParams, run_experiment, run_project


def _sample_params() -> WorkflowParams:
    return WorkflowParams(
        base_task_times={
            "clarify_requirements": 2.0,
            "adam_programming": 1.5,
            "tlf_programming": 1.2,
            "qc_and_reconcile": 1.0,
            "reporting_package": 0.5,
        },
        duration_sigma=0.2,
        miscommunication_prob=0.1,
        late_defect_prob=0.05,
        early_detection_prob=0.5,
        handover_penalty_days=0.5,
        lead_unavailability_prob=0.1,
        base_rework_scale=0.5,
        latent_defect_fix_time=0.7,
    )


def test_run_project_is_deterministic() -> None:
    params = _sample_params()
    resources = {"statisticians": 1, "programmers": 1, "qc": 1}

    result = run_project(params, resources=resources, seed=123)

    assert set(result) == {
        "total_calendar_time",
        "num_rework_cycles",
        "prop_defects_caught_early",
        "num_late_defects",
        "handover_delay",
    }
    assert result["total_calendar_time"] == pytest.approx(6.0549715370361605)
    assert result["num_rework_cycles"] == 0
    assert result["num_late_defects"] == 0
    assert result["handover_delay"] == 0.0


def test_run_experiment_produces_dataframe() -> None:
    params = _sample_params()
    resources = {"statisticians": 1, "programmers": 1, "qc": 1}
    replications = 3
    seed = 999
    rng = np.random.default_rng(seed)
    expected_seeds = [int(s) for s in rng.integers(0, 1_000_000_000, size=replications)]

    df = run_experiment(
        "plaintext",
        params,
        resources=resources,
        replications=replications,
        seed=seed,
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == replications
    assert df["seed"].tolist() == expected_seeds
    assert set(df["condition"]) == {"plaintext"}
    for column in [
        "total_calendar_time",
        "num_rework_cycles",
        "prop_defects_caught_early",
        "num_late_defects",
        "handover_delay",
    ]:
        assert column in df
        assert df[column].ge(0).all()

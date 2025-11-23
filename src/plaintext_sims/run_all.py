"""Run all experiments and collate results."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from plaintext_sims.config import load_config
from plaintext_sims.simpy_experiment import WorkflowParams, bootstrap_difference
from plaintext_sims.simpy_experiment import format_summary_text as format_simpy_summary
from plaintext_sims.simpy_experiment import plot_results as plot_simpy_results
from plaintext_sims.simpy_experiment import run_experiment as run_simpy_experiment
from plaintext_sims.simpy_experiment import summarize_metrics as summarize_simpy_metrics


def main() -> None:
    cfg = load_config()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # SimPy experiment
    sim_cfg = cfg["simpy"]
    sim_metrics = [
        "total_calendar_time",
        "num_rework_cycles",
        "prop_defects_caught_early",
        "num_late_defects",
        "handover_delay",
    ]
    sim_plaintext_params = WorkflowParams(**sim_cfg["params"]["plaintext"])
    sim_mixed_params = WorkflowParams(**sim_cfg["params"]["mixed"])

    sim_plaintext = run_simpy_experiment(
        "plaintext",
        sim_plaintext_params,
        resources=sim_cfg["resources"],
        replications=sim_cfg["replications"],
        seed=sim_cfg["seed"],
    )
    sim_mixed = run_simpy_experiment(
        "mixed",
        sim_mixed_params,
        resources=sim_cfg["resources"],
        replications=sim_cfg["replications"],
        seed=sim_cfg["seed"] + 1,
    )
    sim_results = pd.concat([sim_plaintext, sim_mixed], ignore_index=True)
    sim_output_dir = results_dir
    sim_results.to_csv(sim_output_dir / "simpy_results.csv", index=False)
    sim_summary_df = summarize_simpy_metrics(sim_results, metrics=sim_metrics)
    sim_summary_df.to_csv(sim_output_dir / "simpy_summary.csv", index=False)
    plot_simpy_results(sim_results, output_dir=sim_output_dir, metrics=sim_metrics)

    sim_boot = [
        bootstrap_difference(
            sim_results,
            metric=m,
            condition_a="plaintext",
            condition_b="mixed",
            reps=sim_cfg["bootstrap_reps"],
            rng=np.random.default_rng(sim_cfg["seed"]),
        )
        for m in sim_metrics
    ]
    pd.DataFrame(sim_boot).to_csv(
        sim_output_dir / "simpy_bootstrap_effects.csv", index=False
    )
    sim_summary_text = format_simpy_summary(sim_results)
    (sim_output_dir / "simpy_summary.txt").write_text(
        sim_summary_text, encoding="utf-8"
    )

    print("SimPy experiment complete. See results/ for outputs.")


if __name__ == "__main__":
    main()

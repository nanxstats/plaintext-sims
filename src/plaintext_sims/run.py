"""Orchestrate the SimPy experiment and write outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from plaintext_sims import (
    WorkflowParams,
    bootstrap_difference,
    format_summary_text,
    load_config,
    plot_results,
    run_experiment,
    summarize_metrics,
)


def main() -> None:
    cfg = load_config()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

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

    sim_plaintext = run_experiment(
        "plaintext",
        sim_plaintext_params,
        resources=sim_cfg["resources"],
        replications=sim_cfg["replications"],
        seed=sim_cfg["seed"],
    )
    sim_mixed = run_experiment(
        "mixed",
        sim_mixed_params,
        resources=sim_cfg["resources"],
        replications=sim_cfg["replications"],
        seed=sim_cfg["seed"] + 1,
    )
    sim_results = pl.concat([sim_plaintext, sim_mixed], how="vertical")
    sim_results.write_csv(results_dir / "simpy_results.csv")

    sim_summary_df = summarize_metrics(sim_results, metrics=sim_metrics)
    sim_summary_df.write_csv(results_dir / "simpy_summary.csv")
    plot_results(sim_results, output_dir=results_dir, metrics=sim_metrics)

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
    pl.DataFrame(sim_boot).write_csv(results_dir / "simpy_bootstrap_effects.csv")
    sim_summary_text = format_summary_text(sim_results)
    (results_dir / "simpy_summary.txt").write_text(sim_summary_text, encoding="utf-8")

    print("SimPy experiment complete. See results/ for outputs.")


if __name__ == "__main__":
    main()

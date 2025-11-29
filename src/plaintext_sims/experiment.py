"""SimPy discrete-event workflow experiment."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import polars as pl
import simpy
from tqdm.auto import tqdm


@dataclass
class WorkflowParams:
    base_task_times: dict[str, float]
    duration_sigma: float
    miscommunication_prob: float
    late_defect_prob: float
    early_detection_prob: float
    handover_penalty_days: float
    lead_unavailability_prob: float
    base_rework_scale: float
    latent_defect_fix_time: float


class TaskSpec(TypedDict):
    resource: str
    needs_lead: bool


class ProjectMetrics(TypedDict):
    total_calendar_time: float
    num_rework_cycles: int
    prop_defects_caught_early: float
    num_late_defects: int
    handover_delay: float


class ExperimentRecord(ProjectMetrics):
    condition: str
    seed: int


def _lognormal_duration(
    mean_days: float, sigma: float, rng: np.random.Generator
) -> float:
    """Sample a positive duration with mild skew."""
    return float(rng.lognormal(mean=np.log(mean_days), sigma=sigma))


def run_project(
    params: WorkflowParams, resources: dict[str, int], seed: int
) -> ProjectMetrics:
    """Run one simulated project and return metrics."""
    rng = np.random.default_rng(seed)
    env = simpy.Environment()

    res = {
        "statistician": simpy.Resource(env, capacity=resources["statisticians"]),
        "programmer": simpy.Resource(env, capacity=resources["programmers"]),
        "qc": simpy.Resource(env, capacity=resources["qc"]),
    }

    task_order: list[str] = [
        "clarify_requirements",
        "adam_programming",
        "tlf_programming",
        "qc_and_reconcile",
        "reporting_package",
    ]
    task_resources: dict[str, TaskSpec] = {
        "clarify_requirements": {"resource": "statistician", "needs_lead": True},
        "adam_programming": {"resource": "programmer", "needs_lead": False},
        "tlf_programming": {"resource": "programmer", "needs_lead": False},
        "qc_and_reconcile": {"resource": "qc", "needs_lead": False},
        "reporting_package": {"resource": "statistician", "needs_lead": True},
    }

    latent_defects = 0
    early_catches = 0
    rework_cycles = 0
    handover_delay = 0.0

    def perform_task(task_name: str) -> Generator[simpy.events.Event, None, None]:
        nonlocal latent_defects, early_catches, rework_cycles, handover_delay
        spec = task_resources[task_name]
        resource_name = spec["resource"]
        with res[resource_name].request() as req:
            yield req
            lead_missing = rng.random() < params.lead_unavailability_prob
            if spec["needs_lead"] and lead_missing:
                delay = params.handover_penalty_days * rng.uniform(0.8, 1.2)
                handover_delay += delay
                yield env.timeout(delay)

            base_mean = params.base_task_times[task_name]
            duration = _lognormal_duration(base_mean, params.duration_sigma, rng)
            yield env.timeout(duration)

            if task_name != "qc_and_reconcile":
                if rng.random() < params.miscommunication_prob:
                    if rng.random() < params.early_detection_prob:
                        early_catches += 1
                        rework_cycles += 1
                        rework_time = _lognormal_duration(
                            base_mean * params.base_rework_scale,
                            params.duration_sigma,
                            rng,
                        )
                        yield env.timeout(rework_time)
                    else:
                        latent_defects += 1
                if rng.random() < params.late_defect_prob:
                    latent_defects += 1

    def qc_stage() -> Generator[simpy.events.Event, None, None]:
        nonlocal latent_defects, rework_cycles
        yield from perform_task("qc_and_reconcile")
        if latent_defects:
            fixes = latent_defects
            for _ in range(fixes):
                rework_cycles += 1
                fix_time = _lognormal_duration(
                    params.latent_defect_fix_time, params.duration_sigma, rng
                )
                yield env.timeout(fix_time)
            latent_defects = 0

    def project_flow() -> Generator[simpy.events.Event, None, None]:
        for name in task_order:
            if name == "qc_and_reconcile":
                yield from qc_stage()
            else:
                yield from perform_task(name)

    env.process(project_flow())
    env.run()

    late_defects = max(0, rework_cycles - early_catches)
    total_defects = early_catches + late_defects
    prop_defects_caught_early = early_catches / total_defects if total_defects else 0.0

    return {
        "total_calendar_time": float(env.now),
        "num_rework_cycles": rework_cycles,
        "prop_defects_caught_early": prop_defects_caught_early,
        "num_late_defects": late_defects,
        "handover_delay": handover_delay,
    }


def run_experiment(
    condition_name: str,
    params: WorkflowParams,
    resources: dict[str, int],
    replications: int,
    seed: int,
) -> pl.DataFrame:
    """Run many replications for one condition."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1_000_000_000, size=replications)
    records: list[ExperimentRecord] = []
    for sim_seed in tqdm(seeds, desc=f"Sim {condition_name}", leave=False):
        metrics = run_project(params, resources=resources, seed=int(sim_seed))
        record: ExperimentRecord = {
            **metrics,
            "condition": condition_name,
            "seed": int(sim_seed),
        }
        records.append(record)
    return pl.DataFrame(records)


__all__ = ["WorkflowParams", "run_project", "run_experiment"]

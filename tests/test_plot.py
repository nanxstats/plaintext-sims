from __future__ import annotations

from pathlib import Path

import pandas as pd

from plaintext_sims.plot import plot_results


def test_plot_results_writes_files(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "condition": ["plaintext", "mixed", "plaintext", "mixed"],
            "total_calendar_time": [10.0, 8.5, 11.0, 9.0],
            "num_late_defects": [1.0, 2.0, 1.2, 2.2],
        }
    )
    metrics = ["total_calendar_time", "num_late_defects"]

    plot_results(df, output_dir=tmp_path, metrics=metrics)

    # Check that the single composed ridgeline plot was created
    assert (tmp_path / "metrics_ridgeline.png").exists()

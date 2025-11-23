"""Plotting utilities using plotnine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    geom_boxplot,
    geom_point,
    geom_text,
    ggplot,
    labs,
    theme,
    theme_bw,
)


def plot_results(df: pd.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    """Create comparative plots with plotnine (ggplot-style)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        plot = (
            ggplot(df, aes(x="condition", y=metric, fill="condition"))
            + geom_boxplot()
            + theme_bw()
            + labs(title=metric.replace("_", " ").title(), x="Condition", y=metric)
            + theme(legend_position="none")
        )
        plot.save(filename=str(output_dir / f"{metric}_boxplot.png"), dpi=300)

    trade = (
        df.groupby("condition")[["total_calendar_time", "num_late_defects"]]
        .mean()
        .reset_index()
    )
    trade_plot = (
        ggplot(
            trade,
            aes(
                x="total_calendar_time",
                y="num_late_defects",
                color="condition",
                label="condition",
            ),
        )
        + geom_point(size=3)
        + geom_text(nudge_x=1.0, nudge_y=0.02, size=9)
        + theme_bw()
        + labs(
            x="Mean total time (days)",
            y="Mean late defects",
            title="Time vs late defects",
        )
    )
    trade_plot.save(
        filename=str(output_dir / "tradeoff_time_vs_late_defects.png"), dpi=300
    )


__all__ = ["plot_results"]

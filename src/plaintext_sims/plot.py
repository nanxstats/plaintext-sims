"""Plotting utilities using plotnine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_line,
    element_text,
    geom_boxplot,
    geom_point,
    geom_text,
    ggplot,
    labs,
    theme,
    theme_classic,
)


def plot_results(df: pd.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    """Create comparative plots with plotnine (ggplot-style)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        plot = (
            ggplot(df, aes(x="condition", y=metric, fill="condition"))
            + geom_boxplot()
            + labs(title=metric.replace("_", " ").title(), x="Condition", y=metric)
            + theme(legend_position="none")
            + theme_classic(base_size=14)
            + theme(
                axis_text=element_text(color="0", size=12),
                axis_ticks=element_line(color="0", linewidth=0.5),
            )
        )
        plot.save(
            filename=str(output_dir / f"{metric}_boxplot.png"), dpi=300, verbose=False
        )

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
        + labs(
            x="Mean total time (days)",
            y="Mean late defects",
            title="Time vs late defects",
        )
        + theme_classic(base_size=14)
        + theme(
            axis_text=element_text(color="0", size=12),
            axis_ticks=element_line(color="0", linewidth=0.5),
        )
    )
    trade_plot.save(
        filename=str(output_dir / "tradeoff_time_vs_late_defects.png"),
        dpi=300,
        verbose=False,
    )


__all__ = ["plot_results"]

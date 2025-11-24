"""Plotting utilities using plotnine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_line,
    element_text,
    geom_violin,
    ggplot,
    labs,
    scale_y_continuous,
    theme,
    theme_classic,
)


def plot_results(df: pd.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    """Create comparative ridgeline plots with plotnine."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create individual ridgeline plots for each metric
    plots = []
    for metric in metrics:
        # Add mean value for fill aesthetic
        df_with_mean = df.copy()
        df_with_mean[f"{metric}_mean"] = df_with_mean.groupby("condition")[
            metric
        ].transform("mean")

        plot = (
            ggplot(df_with_mean, aes(x="condition", y=metric, fill=f"{metric}_mean"))
            + geom_violin(
                position="identity",
                style="right",
                width=1.5,
                color="none",
                trim=False,
                alpha=0.85,
            )
            + coord_flip()
            + labs(title=metric.replace("_", " ").title(), x="", y="")
            + scale_y_continuous(expand=(0, 0.1))
            + theme_classic(base_size=12)
            + theme(
                legend_position="none",
                axis_text=element_text(color="0", size=7),
                axis_ticks=element_line(color="0", linewidth=0.5),
                plot_title=element_text(size=9, face="plain"),
            )
        )
        plots.append(plot)

    # Compose all plots in one column using the / operator
    if len(plots) == 1:
        final_plot = plots[0]
    else:
        final_plot = plots[0]
        for plot in plots[1:]:
            final_plot = final_plot / plot  # type: ignore[assignment]

    # Save the composed plot as a single file
    final_plot.save(
        filename=str(output_dir / "metrics_ridgeline.pdf"),
        dpi=300,
        verbose=False,
        width=6 * len(plots),  # Scale width based on number of plots
        height=6,
    )


__all__ = ["plot_results"]

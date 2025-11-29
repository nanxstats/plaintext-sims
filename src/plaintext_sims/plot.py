"""Plotting utilities using plotnine."""

from __future__ import annotations

from pathlib import Path

import polars as pl
from ggsci import scale_fill_aaas  # type: ignore[import-untyped]
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


def plot_results(df: pl.DataFrame, output_dir: Path, metrics: list[str]) -> None:
    """Create comparative ridgeline plots with plotnine."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create individual ridgeline plots for each metric
    plots = []
    for metric in metrics:
        base_plot = df >> ggplot(aes(x="condition", y=metric, fill="condition"))
        plot = (
            base_plot
            # First layer: filled violins
            + geom_violin(
                position="identity",
                style="right",
                width=1.5,
                color="none",
                trim=False,
                alpha=0.85,
            )
            # Second layer: black outline on top
            + geom_violin(
                position="identity",
                style="right",
                width=1.5,
                color="black",
                fill="none",
                trim=False,
                size=0.2,
            )
            + scale_fill_aaas()
            + coord_flip()
            + labs(title=metric.replace("_", " ").title(), x="", y="")
            + scale_y_continuous(expand=(0.05, 0))
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
        filename=str(output_dir / "metrics_ridgeline.png"),
        dpi=300,
        verbose=False,
        width=6,
        height=4 * len(plots),
    )


__all__ = ["plot_results"]

"""Bokeh visualization for curve matches and mapped test points."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from bokeh.layouts import column
from bokeh.models import HoverTool, Tabs
from bokeh.plotting import figure, output_file, save

try:
    from bokeh.models import TabPanel
except ImportError:  # Compatibility with Bokeh 2.x
    from bokeh.models import Panel as TabPanel

from .curves import ReferenceCurve, TrainingCurve


class Visualizer:
    """Create an interactive HTML report with one tab per training curve."""

    COLORS = ("#d62728", "#1f77b4", "#2ca02c")

    def create_plot(
        self,
        training_curves: list[TrainingCurve],
        reference_curves: list[ReferenceCurve],
        top_three: pd.DataFrame,
        mapping_results: pd.DataFrame,
        output_path: Path | str,
    ) -> None:
        """Save training, Top-3 references, and mapped points as Bokeh HTML."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        reference_lookup = {curve.curve_id: curve for curve in reference_curves}
        tabs: list[TabPanel] = []

        for training_curve in training_curves:
            plot = figure(
                title=f"Cp matching: {training_curve.curve_id}",
                x_axis_label="x/c",
                y_axis_label="Cp",
                width=1000,
                height=560,
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            plot.line(
                training_curve.x_values,
                training_curve.cp_values,
                line_width=3,
                color="#111111",
                legend_label=f"Training: {training_curve.curve_id}",
            )

            candidates = top_three.loc[
                top_three["training_curve_id"] == training_curve.curve_id
            ].sort_values("rank")
            for candidate in candidates.itertuples(index=False):
                reference = reference_lookup[str(candidate.reference_curve_id)]
                rank = int(candidate.rank)
                plot.line(
                    reference.x_values,
                    reference.cp_values,
                    line_width=2.5 if rank == 1 else 1.8,
                    line_dash="solid" if rank == 1 else "dashed",
                    color=self.COLORS[rank - 1],
                    alpha=0.95 if rank == 1 else 0.75,
                    legend_label=(
                        f"Rank {rank}: {reference.curve_id} "
                        f"(SSE={candidate.sse:.5g})"
                    ),
                )

            points = mapping_results.loc[
                mapping_results["training_curve_id"] == training_curve.curve_id
            ]
            if not points.empty:
                accepted_points = points.loc[points["accepted"]]
                rejected_points = points.loc[~points["accepted"]]
                if not accepted_points.empty:
                    plot.scatter(
                        accepted_points["x_test"],
                        accepted_points["cp_test"],
                        size=11,
                        marker="circle",
                        color="#9467bd",
                        legend_label="Accepted test point",
                    )
                if not rejected_points.empty:
                    plot.scatter(
                        rejected_points["x_test"],
                        rejected_points["cp_test"],
                        size=13,
                        marker="x",
                        line_width=3,
                        color="#ff7f0e",
                        legend_label="Rejected test point",
                    )
                plot.scatter(
                    points["x_test"],
                    points["reference_cp_interpolated"],
                    size=9,
                    marker="diamond",
                    color="#17becf",
                    legend_label="Interpolated reference Cp",
                )

            plot.add_tools(
                HoverTool(
                    tooltips=[
                        ("x/c", "$x{0.0000}"),
                        ("Cp", "$y{0.0000}"),
                    ]
                )
            )
            plot.legend.location = "top_right"
            plot.legend.click_policy = "hide"
            tabs.append(TabPanel(child=plot, title=training_curve.curve_id))

        if not tabs:
            raise ValueError("At least one training curve is required for plotting.")

        output_file(destination, title="NACA Cp Curve Matching")
        save(column(Tabs(tabs=tabs)))

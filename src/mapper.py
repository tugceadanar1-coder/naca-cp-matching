"""Check test Cp points against the selected reference curves."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .curves import ReferenceCurve, TestPoint


class TestPointMapper:
    """Map each test point to the best reference for its training curve."""

    @staticmethod
    def _interpolate_reference(reference: ReferenceCurve, x_test: float) -> float:
        """Find the reference Cp value at the test point x/c location."""
        interpolation_data = pd.DataFrame(
            {"x": reference.x_values, "cp": reference.cp_values}
        )
        interpolation_data = (
            interpolation_data.groupby("x", as_index=False)["cp"].mean().sort_values("x")
        )
        x_values = interpolation_data["x"].to_numpy(dtype=float)
        cp_values = interpolation_data["cp"].to_numpy(dtype=float)

        if x_test < x_values[0] or x_test > x_values[-1]:
            raise ValueError(
                f"Test x={x_test} is outside reference curve "
                f"'{reference.curve_id}' range [{x_values[0]}, {x_values[-1]}]."
            )
        return float(np.interp(x_test, x_values, cp_values))

    def map_points(
        self,
        test_points: list[TestPoint],
        reference_curves: list[ReferenceCurve],
        best_matches: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate delta Cp and decide if each test point is accepted."""
        required_best_columns = {
            "training_curve_id",
            "best_reference_curve_id",
            "max_training_deviation",
        }
        if not required_best_columns.issubset(best_matches.columns):
            raise ValueError("best_matches is missing columns needed for point mapping.")

        references = {curve.curve_id: curve for curve in reference_curves}
        match_lookup = best_matches.set_index("training_curve_id").to_dict("index")

        rows: list[dict[str, float | str | bool]] = []
        for point in test_points:
            if point.training_curve_id not in match_lookup:
                raise ValueError(
                    f"No best match exists for '{point.training_curve_id}'."
                )
            match = match_lookup[point.training_curve_id]
            reference_id = str(match["best_reference_curve_id"])
            if reference_id not in references:
                raise ValueError(f"Best reference curve '{reference_id}' was not loaded.")

            reference_cp = self._interpolate_reference(
                references[reference_id],
                point.x_test,
            )
            delta_cp = abs(point.cp_test - reference_cp)
            max_training_deviation = float(match["max_training_deviation"])
            acceptance_limit = max_training_deviation * math.sqrt(2.0)

            # The point is accepted when its Cp difference is inside the limit.
            rows.append(
                {
                    "training_curve_id": point.training_curve_id,
                    "best_reference_curve_id": reference_id,
                    "x_test": point.x_test,
                    "cp_test": point.cp_test,
                    "reference_cp_interpolated": reference_cp,
                    "delta_cp": delta_cp,
                    "max_training_deviation": max_training_deviation,
                    "acceptance_limit": acceptance_limit,
                    "accepted": bool(delta_cp <= acceptance_limit),
                }
            )

        return pd.DataFrame(rows)

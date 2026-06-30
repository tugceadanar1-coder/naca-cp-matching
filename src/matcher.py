"""Compare Cp curves using least-squares error values."""

import numpy as np
import pandas as pd

from .curves import ReferenceCurve, TrainingCurve


class CurveMatcher:
    """Compare each training Cp curve with all reference Cp curves."""

    @staticmethod
    def _validated_difference(
        training_cp: np.ndarray,
        reference_cp: np.ndarray,
    ) -> np.ndarray:
        """Check two Cp arrays before calculating their difference."""
        training = np.asarray(training_cp, dtype=float)
        reference = np.asarray(reference_cp, dtype=float)
        if training.ndim != 1 or reference.ndim != 1:
            raise ValueError("Metric inputs must be one-dimensional arrays.")
        if len(training) == 0:
            raise ValueError("Metric inputs cannot be empty.")
        if training.shape != reference.shape:
            raise ValueError("Training and reference Cp arrays must have equal shapes.")
        if not np.all(np.isfinite(training)) or not np.all(np.isfinite(reference)):
            raise ValueError("Metric inputs must contain only finite values.")
        return training - reference

    @staticmethod
    def _check_matching_x_values(
        training_curve: TrainingCurve,
        reference_curve: ReferenceCurve,
    ) -> None:
        """Make sure two curves are compared at the same x/c positions."""
        same_length = len(training_curve.x_values) == len(reference_curve.x_values)
        if not same_length:
            raise ValueError(
                "Training and reference curves must use the same x/c values."
            )

        same_values = np.allclose(
            training_curve.x_values,
            reference_curve.x_values,
            rtol=0.0,
            atol=1e-12,
        )
        if not same_values:
            raise ValueError(
                "Training and reference curves must use the same x/c values."
            )

    @classmethod
    def calculate_sse(
        cls,
        training_cp: np.ndarray,
        reference_cp: np.ndarray,
    ) -> float:
        """Calculate SSE, the main value used to choose the closest curve."""
        difference = cls._validated_difference(training_cp, reference_cp)
        return float(np.sum(np.square(difference)))

    @classmethod
    def calculate_rmse(
        cls,
        training_cp: np.ndarray,
        reference_cp: np.ndarray,
    ) -> float:
        """Calculate RMSE, which shows the typical Cp error size."""
        difference = cls._validated_difference(training_cp, reference_cp)
        return float(np.sqrt(np.mean(np.square(difference))))

    @classmethod
    def calculate_max_delta_cp(
        cls,
        training_cp: np.ndarray,
        reference_cp: np.ndarray,
    ) -> float:
        """Find the largest absolute Cp difference for two curves."""
        difference = cls._validated_difference(training_cp, reference_cp)
        return float(np.max(np.abs(difference)))

    @staticmethod
    def calculate_confidence_margin(best_sse: float, second_best_sse: float) -> float:
        """Return how far the second-best SSE is from the best SSE."""
        return float(second_best_sse - best_sse)

    @staticmethod
    def calculate_relative_confidence_margin(
        best_sse: float,
        second_best_sse: float,
    ) -> float:
        """Return the confidence margin divided by the best SSE."""
        margin = second_best_sse - best_sse
        if best_sse == 0:
            if margin == 0:
                return 0.0
            return float("inf")
        return float(margin / best_sse)

    def compare_all(
        self,
        training_curves: list[TrainingCurve],
        reference_curves: list[ReferenceCurve],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compare every training curve with every reference curve."""
        if not training_curves:
            raise ValueError("At least one training curve is required.")
        if not reference_curves:
            raise ValueError("At least one reference curve is required.")

        rows = []
        for training_curve in training_curves:
            for reference_curve in reference_curves:
                self._check_matching_x_values(training_curve, reference_curve)

                sse = self.calculate_sse(
                    training_curve.cp_values,
                    reference_curve.cp_values,
                )
                rmse = self.calculate_rmse(
                    training_curve.cp_values,
                    reference_curve.cp_values,
                )
                max_delta_cp = self.calculate_max_delta_cp(
                    training_curve.cp_values,
                    reference_curve.cp_values,
                )

                row = {
                    "training_curve_id": training_curve.curve_id,
                    "reference_curve_id": reference_curve.curve_id,
                    "sse": sse,
                    "rmse": rmse,
                    "max_delta_cp": max_delta_cp,
                }
                rows.append(row)

        # Ranking is based only on SSE. Smaller SSE means a closer curve match.
        audit = pd.DataFrame(rows)
        audit = audit.sort_values(
            ["training_curve_id", "sse", "reference_curve_id"],
            kind="stable",
        ).reset_index(drop=True)
        audit["rank"] = audit.groupby("training_curve_id").cumcount() + 1
        audit["is_best_match"] = audit["rank"].eq(1)

        top_three = audit.loc[audit["rank"] <= 3].copy().reset_index(drop=True)
        best_match_rows = []

        for training_curve in training_curves:
            ranked_matches = audit.loc[
                audit["training_curve_id"] == training_curve.curve_id
            ].sort_values("rank")

            best_row = ranked_matches.iloc[0]
            if len(ranked_matches) > 1:
                second_best_row = ranked_matches.iloc[1]
                second_best_reference = second_best_row["reference_curve_id"]
                second_best_sse = float(second_best_row["sse"])
                confidence_margin = self.calculate_confidence_margin(
                    float(best_row["sse"]),
                    second_best_sse,
                )
                relative_confidence_margin = (
                    self.calculate_relative_confidence_margin(
                        float(best_row["sse"]),
                        second_best_sse,
                    )
                )
            else:
                second_best_reference = ""
                second_best_sse = np.nan
                confidence_margin = np.nan
                relative_confidence_margin = np.nan

            best_match_rows.append(
                {
                    "training_curve_id": training_curve.curve_id,
                    "best_reference_curve_id": best_row["reference_curve_id"],
                    "sse": float(best_row["sse"]),
                    "rmse": float(best_row["rmse"]),
                    "max_training_deviation": float(best_row["max_delta_cp"]),
                    "second_best_reference_curve_id": second_best_reference,
                    "second_best_sse": second_best_sse,
                    "confidence_margin": confidence_margin,
                    "relative_confidence_margin": relative_confidence_margin,
                }
            )

        best_matches = pd.DataFrame(best_match_rows)
        return audit, top_three, best_matches

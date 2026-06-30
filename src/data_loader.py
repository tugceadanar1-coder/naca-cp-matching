"""Load the cleaned Cp input files."""

from pathlib import Path

import numpy as np
import pandas as pd

from .curves import Curve, ReferenceCurve, TestPoint, TrainingCurve


class DataLoader:
    """Read training curves, reference curves, and test points."""

    def __init__(self, data_directory: Path | str) -> None:
        self.data_directory = Path(data_directory)
        self.training_path = self.data_directory / "training_curves.csv"
        self.reference_path = self.data_directory / "reference_curves.csv"
        self.test_points_path = self.data_directory / "test_points.csv"

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        """Read one CSV file and check that it contains data."""
        if not path.exists():
            raise FileNotFoundError(
                f"Required cleaned data file was not found: {path}."
            )
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Data file is empty: {path}") from exc
        except Exception as exc:
            raise ValueError(f"Could not read CSV file '{path}': {exc}") from exc

        if frame.empty:
            raise ValueError(f"Data file contains no rows: {path}")
        return frame

    @staticmethod
    def _validate_numeric_columns(
        frame: pd.DataFrame,
        columns: list[str],
        file_label: str,
    ) -> pd.DataFrame:
        """Convert selected columns to numeric values."""
        cleaned = frame.copy()
        for column in columns:
            try:
                cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Column '{column}' in {file_label} must contain only numbers."
                ) from exc

        if cleaned[columns].isna().any().any():
            raise ValueError(f"{file_label} contains missing numeric values.")
        values = cleaned[columns].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{file_label} contains non-finite numeric values.")
        return cleaned

    def _load_curves(
        self,
        path: Path,
        required_prefix: str,
        curve_class: type,
    ) -> tuple[pd.DataFrame, list[Curve]]:
        """Load a curve table and create Curve objects."""
        frame = self._read_csv(path)
        if "x" not in frame.columns:
            raise ValueError(f"{path.name} must contain an 'x' column.")

        curve_columns = []
        for column in frame.columns:
            if column.startswith(required_prefix):
                curve_columns.append(column)

        if not curve_columns:
            raise ValueError(
                f"{path.name} must contain at least one '{required_prefix}...' column."
            )

        unexpected = []
        for column in frame.columns:
            if column != "x" and column not in curve_columns:
                unexpected.append(column)

        if unexpected:
            raise ValueError(
                f"{path.name} contains unexpected columns: {', '.join(unexpected)}"
            )

        numeric_columns = ["x"] + curve_columns
        frame = self._validate_numeric_columns(
            frame,
            numeric_columns,
            path.name,
        )
        x_values = frame["x"].to_numpy(dtype=float)

        curves = []
        for column in curve_columns:
            curve = curve_class(
                curve_id=column,
                x_values=x_values,
                cp_values=frame[column].to_numpy(dtype=float),
            )
            curves.append(curve)

        return frame, curves

    def load_training_curves(self) -> tuple[pd.DataFrame, list[TrainingCurve]]:
        """Load the training Cp curves."""
        return self._load_curves(
            self.training_path,
            required_prefix="training_",
            curve_class=TrainingCurve,
        )

    def load_reference_curves(self) -> tuple[pd.DataFrame, list[ReferenceCurve]]:
        """Load the reference Cp curves."""
        return self._load_curves(
            self.reference_path,
            required_prefix="reference_",
            curve_class=ReferenceCurve,
        )

    def load_test_points(self) -> tuple[pd.DataFrame, list[TestPoint]]:
        """Load the test Cp points."""
        frame = self._read_csv(self.test_points_path)
        required_columns = ["x_test", "cp_test", "training_curve_id"]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                f"{self.test_points_path.name} is missing columns: {', '.join(missing)}"
            )
        unexpected = [
            column for column in frame.columns if column not in required_columns
        ]
        if unexpected:
            raise ValueError(
                f"{self.test_points_path.name} contains unexpected columns: "
                f"{', '.join(unexpected)}"
            )

        frame = self._validate_numeric_columns(
            frame,
            ["x_test", "cp_test"],
            self.test_points_path.name,
        )
        if frame["training_curve_id"].isna().any():
            raise ValueError("test_points.csv contains missing training_curve_id values.")
        frame["training_curve_id"] = frame["training_curve_id"].astype(str)

        points = [
            TestPoint(
                x_test=float(row.x_test),
                cp_test=float(row.cp_test),
                training_curve_id=str(row.training_curve_id),
            )
            for row in frame.itertuples(index=False)
        ]
        return frame, points

    @staticmethod
    def validate_matching_x_values(
        training_curves: list[TrainingCurve],
        reference_curves: list[ReferenceCurve],
    ) -> None:
        """Check that all curves use the same x/c grid."""
        if not training_curves or not reference_curves:
            raise ValueError("Training and reference curve collections cannot be empty.")

        expected_x = training_curves[0].x_values
        all_curves: list[Curve] = [*training_curves, *reference_curves]
        for curve in all_curves:
            if len(curve.x_values) != len(expected_x) or not np.allclose(
                curve.x_values,
                expected_x,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Training and reference curves must have matching x-values. "
                    f"Curve '{curve.curve_id}' does not match."
                )

    def load_all(
        self,
    ) -> tuple[
        list[TrainingCurve],
        list[ReferenceCurve],
        list[TestPoint],
    ]:
        """Load all input data and run basic consistency checks."""
        _, training_curves = self.load_training_curves()
        _, reference_curves = self.load_reference_curves()
        _, test_points = self.load_test_points()
        self.validate_matching_x_values(training_curves, reference_curves)

        training_ids = {curve.curve_id for curve in training_curves}
        unknown_ids = sorted(
            {
                point.training_curve_id
                for point in test_points
                if point.training_curve_id not in training_ids
            }
        )
        if unknown_ids:
            raise ValueError(
                "test_points.csv refers to unknown training curves: "
                + ", ".join(unknown_ids)
            )
        return training_curves, reference_curves, test_points

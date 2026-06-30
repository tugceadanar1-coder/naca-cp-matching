"""Data classes used for Cp curves and test points."""

import numpy as np


class Curve:
    """Represent a single pressure coefficient (Cp) curve."""

    def __init__(
        self,
        curve_id: str,
        x_values: np.ndarray,
        cp_values: np.ndarray,
    ) -> None:
        self.curve_id = str(curve_id)
        self.x_values = np.asarray(x_values, dtype=float).copy()
        self.cp_values = np.asarray(cp_values, dtype=float).copy()
        self._validate()

    def _validate(self) -> None:
        """Validate the curve data."""
        if self.x_values.ndim != 1 or self.cp_values.ndim != 1:
            raise ValueError(f"Curve '{self.curve_id}' must contain one-dimensional arrays.")
        if len(self.x_values) == 0:
            raise ValueError(f"Curve '{self.curve_id}' is empty.")
        if len(self.x_values) != len(self.cp_values):
            raise ValueError(
                f"Curve '{self.curve_id}' has different x and Cp array lengths."
            )
        if not np.all(np.isfinite(self.x_values)):
            raise ValueError(f"Curve '{self.curve_id}' contains invalid x values.")
        if not np.all(np.isfinite(self.cp_values)):
            raise ValueError(f"Curve '{self.curve_id}' contains invalid Cp values.")

    def __len__(self) -> int:
        """Return the number of data points."""
        return len(self.x_values)


class TrainingCurve(Curve):
    """Training Cp curve."""


class ReferenceCurve(Curve):
    """Reference Cp curve."""


class TestPoint:
    """Represent a single test Cp point."""

    def __init__(self, x_test: float, cp_test: float, training_curve_id: str) -> None:
        self.x_test = float(x_test)
        self.cp_test = float(cp_test)
        self.training_curve_id = str(training_curve_id)
        self._validate()

    def _validate(self) -> None:
        """Validate the test point data."""
        if not np.isfinite(self.x_test) or not np.isfinite(self.cp_test):
            raise ValueError("Test points must contain finite x_test and cp_test values.")
        if not self.training_curve_id.strip():
            raise ValueError("A test point must have a training_curve_id.")

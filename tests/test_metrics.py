"""Unit tests for the Cp curve matching project."""

import math
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.curves import ReferenceCurve, TestPoint, TrainingCurve
from src.mapper import TestPointMapper
from src.matcher import CurveMatcher


class TestCurveMatcherMetrics(unittest.TestCase):
    """Test the main mathematical calculations."""

    def test_sse_calculation(self) -> None:
        """SSE is the sum of squared point-by-point differences."""
        training = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.0, 1.0, 5.0])

        self.assertAlmostEqual(
            CurveMatcher.calculate_sse(training, reference),
            5.0,
        )

    def test_rmse_calculation(self) -> None:
        """RMSE is the square root of SSE divided by sample count."""
        training = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.0, 1.0, 5.0])

        self.assertAlmostEqual(
            CurveMatcher.calculate_rmse(training, reference),
            np.sqrt(5.0 / 3.0),
        )

    def test_max_delta_cp_calculation(self) -> None:
        """Maximum delta Cp uses the largest absolute difference."""
        training = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.0, 1.0, 5.0])

        self.assertAlmostEqual(
            CurveMatcher.calculate_max_delta_cp(training, reference),
            2.0,
        )

    def test_confidence_margin_calculation(self) -> None:
        """Confidence margin is second-best SSE minus best SSE."""
        self.assertAlmostEqual(
            CurveMatcher.calculate_confidence_margin(0.01, 0.25),
            0.24,
        )
        self.assertAlmostEqual(
            CurveMatcher.calculate_relative_confidence_margin(0.01, 0.25),
            24.0,
        )

    def test_empty_input_arrays_raise_value_error(self) -> None:
        """Metric functions should reject empty Cp arrays."""
        training = np.array([])
        reference = np.array([])

        with self.assertRaises(ValueError):
            CurveMatcher.calculate_sse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_rmse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_max_delta_cp(training, reference)

    def test_different_length_arrays_raise_value_error(self) -> None:
        """Metric functions should reject arrays with different lengths."""
        training = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.0, 2.0])

        with self.assertRaises(ValueError):
            CurveMatcher.calculate_sse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_rmse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_max_delta_cp(training, reference)

    def test_nan_values_raise_value_error(self) -> None:
        """Metric functions should reject Cp arrays containing NaN."""
        training = np.array([1.0, np.nan, 3.0])
        reference = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            CurveMatcher.calculate_sse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_rmse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_max_delta_cp(training, reference)

    def test_infinity_values_raise_value_error(self) -> None:
        """Metric functions should reject Cp arrays containing infinity."""
        training = np.array([1.0, 2.0, np.inf])
        reference = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            CurveMatcher.calculate_sse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_rmse(training, reference)
        with self.assertRaises(ValueError):
            CurveMatcher.calculate_max_delta_cp(training, reference)


class TestCurveMatcherRanking(unittest.TestCase):
    """Test best-match and Top-3 selection."""

    def make_example_curves(self):
        x_values = np.array([0.0, 0.5, 1.0])
        training_curves = [
            TrainingCurve("training_1", x_values, np.array([0.0, 1.0, 0.0]))
        ]
        reference_curves = [
            ReferenceCurve("reference_far", x_values, np.array([1.0, 2.0, 1.0])),
            ReferenceCurve("reference_best", x_values, np.array([0.0, 1.1, 0.0])),
            ReferenceCurve("reference_middle", x_values, np.array([0.0, 1.5, 0.0])),
            ReferenceCurve(
                "reference_farthest",
                x_values,
                np.array([3.0, 3.0, 3.0]),
            ),
        ]
        return training_curves, reference_curves

    def test_best_match_selection(self) -> None:
        """The reference with the smallest SSE is selected as the best match."""
        training_curves, reference_curves = self.make_example_curves()

        matcher = CurveMatcher()
        error_audit, top_three, best_matches = matcher.compare_all(
            training_curves,
            reference_curves,
        )

        self.assertEqual(
            best_matches.loc[0, "best_reference_curve_id"],
            "reference_best",
        )
        self.assertEqual(len(error_audit), 4)
        self.assertEqual(len(top_three), 3)

    def test_second_best_reference_selection(self) -> None:
        """The second-best reference is the candidate with rank 2."""
        training_curves, reference_curves = self.make_example_curves()

        matcher = CurveMatcher()
        _, _, best_matches = matcher.compare_all(training_curves, reference_curves)

        self.assertEqual(
            best_matches.loc[0, "second_best_reference_curve_id"],
            "reference_middle",
        )
        self.assertAlmostEqual(best_matches.loc[0, "sse"], 0.01)
        self.assertAlmostEqual(best_matches.loc[0, "second_best_sse"], 0.25)
        self.assertAlmostEqual(best_matches.loc[0, "confidence_margin"], 0.24)
        self.assertAlmostEqual(
            best_matches.loc[0, "relative_confidence_margin"],
            24.0,
        )

    def test_top_three_ranking_order(self) -> None:
        """Top-3 candidates are ordered by increasing SSE."""
        training_curves, reference_curves = self.make_example_curves()

        matcher = CurveMatcher()
        _, top_three, _ = matcher.compare_all(training_curves, reference_curves)

        self.assertEqual(
            top_three["reference_curve_id"].tolist(),
            ["reference_best", "reference_middle", "reference_far"],
        )
        self.assertEqual(top_three["rank"].tolist(), [1, 2, 3])
        self.assertTrue(top_three["sse"].is_monotonic_increasing)


class TestGridConsistency(unittest.TestCase):
    """Test that processed data files use the same x/c grid."""

    def test_training_and_reference_csv_files_use_same_x_grid(self) -> None:
        """Training, reference, and x_grid CSV files should line up exactly."""
        project_directory = Path(__file__).resolve().parents[1]
        data_directory = project_directory / "data"

        x_grid = pd.read_csv(data_directory / "x_grid.csv")
        training = pd.read_csv(data_directory / "training_curves.csv")
        reference = pd.read_csv(data_directory / "reference_curves.csv")

        self.assertEqual(x_grid["x"].tolist(), training["x"].tolist())
        self.assertEqual(x_grid["x"].tolist(), reference["x"].tolist())


class TestCurveMatcherValidation(unittest.TestCase):
    """Test input checks that protect the least-squares comparison."""

    def test_compare_all_rejects_mismatched_x_values(self) -> None:
        """Curves cannot be compared when their x/c grids are different."""
        training_curve = TrainingCurve(
            "training_1",
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        reference_curve = ReferenceCurve(
            "reference_1",
            np.array([0.0, 0.4, 1.0]),
            np.array([0.0, 1.0, 0.0]),
        )

        matcher = CurveMatcher()
        with self.assertRaises(ValueError):
            matcher.compare_all([training_curve], [reference_curve])


class TestPointMapperRule(unittest.TestCase):
    """Test the Cp test-point acceptance rule."""

    def make_mapping_inputs(self, cp_test: float):
        """Create a small mapper example with one reference curve."""
        reference_curve = ReferenceCurve(
            "reference_1",
            np.array([0.0, 1.0]),
            np.array([1.0, 1.0]),
        )
        point = TestPoint(
            x_test=0.5,
            cp_test=cp_test,
            training_curve_id="training_1",
        )
        best_matches = pd.DataFrame(
            [
                {
                    "training_curve_id": "training_1",
                    "best_reference_curve_id": "reference_1",
                    "max_training_deviation": 0.5,
                }
            ]
        )
        return [point], [reference_curve], best_matches

    def test_mapper_accepts_point_at_acceptance_limit(self) -> None:
        """A point is accepted when delta Cp equals the limit."""
        acceptance_limit = 0.5 * math.sqrt(2.0)
        test_points, reference_curves, best_matches = self.make_mapping_inputs(
            cp_test=1.0 + acceptance_limit,
        )

        mapper = TestPointMapper()
        result = mapper.map_points(test_points, reference_curves, best_matches)

        self.assertAlmostEqual(result.loc[0, "delta_cp"], acceptance_limit)
        self.assertTrue(bool(result.loc[0, "accepted"]))

    def test_mapper_rejects_point_above_acceptance_limit(self) -> None:
        """A point is rejected when delta Cp is larger than the limit."""
        acceptance_limit = 0.5 * math.sqrt(2.0)
        test_points, reference_curves, best_matches = self.make_mapping_inputs(
            cp_test=1.0 + acceptance_limit + 0.001,
        )

        mapper = TestPointMapper()
        result = mapper.map_points(test_points, reference_curves, best_matches)

        self.assertGreater(result.loc[0, "delta_cp"], acceptance_limit)
        self.assertFalse(bool(result.loc[0, "accepted"]))


if __name__ == "__main__":
    unittest.main()

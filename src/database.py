"""Save project results to SQLite."""

import sqlite3
from pathlib import Path

import pandas as pd

from .curves import Curve, ReferenceCurve, TrainingCurve


class DatabaseManager:
    """Handle the SQLite database."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)

    @staticmethod
    def _curves_to_long_table(curves: list[Curve]) -> pd.DataFrame:
        """Convert curves into rows for SQLite."""
        rows = []
        for curve in curves:
            for x_value, cp_value in zip(curve.x_values, curve.cp_values):
                rows.append(
                    {
                        "curve_id": curve.curve_id,
                        "x": x_value,
                        "cp": cp_value,
                    }
                )
        return pd.DataFrame(rows, columns=["curve_id", "x", "cp"])

    def save_all(
        self,
        training_curves: list[TrainingCurve],
        reference_curves: list[ReferenceCurve],
        error_audit: pd.DataFrame,
        best_matches: pd.DataFrame,
        mapping_results: pd.DataFrame,
    ) -> None:
        """Save all project tables."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        training_table = self._curves_to_long_table(training_curves)
        reference_table = self._curves_to_long_table(reference_curves)

        try:
            with sqlite3.connect(self.database_path) as connection:
                training_table.to_sql(
                    "training_curves",
                    connection,
                    if_exists="replace",
                    index=False,
                )
                reference_table.to_sql(
                    "reference_curves",
                    connection,
                    if_exists="replace",
                    index=False,
                )
                error_audit.to_sql(
                    "error_audit",
                    connection,
                    if_exists="replace",
                    index=False,
                )
                best_matches.to_sql(
                    "best_matches",
                    connection,
                    if_exists="replace",
                    index=False,
                )
                mapping_results.to_sql(
                    "mapping_results",
                    connection,
                    if_exists="replace",
                    index=False,
                )
        except sqlite3.Error as exc:
            raise RuntimeError(
                f"Could not save SQLite database '{self.database_path}': {exc}"
            ) from exc

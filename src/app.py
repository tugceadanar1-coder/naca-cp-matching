"""Run the NACA 0012 Cp curve matching workflow."""

from __future__ import annotations

from pathlib import Path

from .data_loader import DataLoader
from .database import DatabaseManager
from .mapper import TestPointMapper
from .matcher import CurveMatcher
from .visualizer import Visualizer


class AssignmentApp:
    """Coordinate data loading, curve matching, mapping, saving, and plotting."""

    def __init__(self, project_directory: Path | str | None = None) -> None:
        self.project_directory = (
            Path(project_directory)
            if project_directory is not None
            else Path(__file__).resolve().parents[1]
        )
        self.data_directory = self.project_directory / "data"
        self.results_directory = self.project_directory / "results"

    def run(self) -> dict[str, Path]:
        """Execute the project workflow and return the generated result file paths."""
        self.results_directory.mkdir(parents=True, exist_ok=True)

        loader = DataLoader(self.data_directory)
        training_curves, reference_curves, test_points = loader.load_all()

        matcher = CurveMatcher()
        error_audit, top_three, best_matches = matcher.compare_all(
            training_curves,
            reference_curves,
        )

        mapper = TestPointMapper()
        mapping_results = mapper.map_points(
            test_points,
            reference_curves,
            best_matches,
        )

        output_paths = {
            "database": self.results_directory / "naca_cp_matching.sqlite",
            "error_audit": self.results_directory / "error_audit.csv",
            "best_matches": self.results_directory / "best_matches.csv",
            "mapping_results": self.results_directory / "mapping_results.csv",
            "plot": self.results_directory / "cp_matching_plot.html",
        }

        error_audit.to_csv(output_paths["error_audit"], index=False)
        best_matches.to_csv(output_paths["best_matches"], index=False)
        mapping_results.to_csv(output_paths["mapping_results"], index=False)

        database = DatabaseManager(output_paths["database"])
        database.save_all(
            training_curves,
            reference_curves,
            error_audit,
            best_matches,
            mapping_results,
        )

        visualizer = Visualizer()
        visualizer.create_plot(
            training_curves,
            reference_curves,
            top_three,
            mapping_results,
            output_paths["plot"],
        )
        return output_paths

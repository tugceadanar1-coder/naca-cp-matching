"""Prepare Gregory and O'Reilly digitized Cp curves as reference data.

The input files are WebPlotDigitizer CSV exports in data/raw/source_curves/.
The x-axis in Gregory and O'Reilly Figure 4 is shifted, so this script first
converts x_plot back to the real x/c value:

    x/c = x_plot - 0.05 * alpha

The final reference curves are interpolated onto the shared x/c grid stored in
data/x_grid.csv. No Cp values are extrapolated outside the digitized data.
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
RAW_DIRECTORY = DATA_DIRECTORY / "raw"
SOURCE_CURVES_DIRECTORY = RAW_DIRECTORY / "source_curves"
SOURCE_PDF = RAW_DIRECTORY / "gregory_oreilly_1970.pdf"
DEFAULT_OUTPUT_FILE = DATA_DIRECTORY / "reference_curves.csv"
DEFAULT_X_GRID_FILE = DATA_DIRECTORY / "x_grid.csv"
NUMBER_OF_REFERENCE_CURVES = 50


def find_digitized_csv_files(digitized_directory: Path) -> list[Path]:
    """Find the digitized Gregory CSV files."""
    csv_files = []

    if not digitized_directory.exists():
        raise FileNotFoundError(
            f"Digitized data directory was not found: {digitized_directory}"
        )

    # Only files in source_curves are used for the reference library.
    for path in sorted(digitized_directory.glob("*.csv")):
        csv_files.append(path)

    if not csv_files:
        raise FileNotFoundError(
            "No digitized CSV files were found in data/raw/source_curves/. "
            "Export Figure 4 curves from WebPlotDigitizer first, then rerun this script."
        )

    return csv_files


def extract_alpha_from_filename(csv_path: Path) -> float:
    """Read alpha from a filename such as cp_alpha_6.csv."""
    match = re.search(
        r"alpha_([-+]?\d+(?:[._]\d+)?)",
        csv_path.stem,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Could not extract alpha from filename '{csv_path.name}'. "
            "Use names such as cp_alpha_0.csv, cp_alpha_6.csv, or cp_alpha_16_5.csv."
        )

    alpha_text = match.group(1).replace("_", ".")
    return float(alpha_text)


def parse_number(text: str) -> float:
    """Read a number that may use a comma decimal mark."""
    return float(str(text).strip().replace(",", "."))


def first_line_has_numeric_data(csv_path: Path) -> bool:
    """Check if the first CSV row already contains numbers."""
    with csv_path.open("r", encoding="utf-8-sig") as file:
        first_line = file.readline().strip()

    if ";" in first_line:
        pieces = first_line.split(";")
    else:
        pieces = first_line.split(",")

    if len(pieces) < 2:
        return False

    try:
        parse_number(pieces[0])
        parse_number(pieces[1])
    except ValueError:
        return False

    return True


def read_digitized_csv(csv_path: Path) -> pd.DataFrame:
    """Read one WebPlotDigitizer CSV file."""
    has_numeric_first_line = first_line_has_numeric_data(csv_path)

    try:
        if has_numeric_first_line:
            with csv_path.open("r", encoding="utf-8-sig") as file:
                first_line = file.readline()

            if ";" in first_line:
                return pd.read_csv(csv_path, sep=";", decimal=",", header=None)

            return pd.read_csv(csv_path, sep=",", decimal=".", header=None)

        try:
            return pd.read_csv(csv_path, sep=";", decimal=",")
        except Exception:
            return pd.read_csv(csv_path)
    except Exception as exc:
        raise ValueError(f"Could not read digitized CSV file '{csv_path}': {exc}") from exc


def find_column(frame: pd.DataFrame, possible_names: list[str]) -> str | None:
    """Find a column using common x and Cp column names."""
    cleaned_names = {}
    for column in frame.columns:
        simple_name = str(column).strip().lower().replace(" ", "").replace("_", "")
        cleaned_names[simple_name] = column

    for name in possible_names:
        simple_name = name.strip().lower().replace(" ", "").replace("_", "")
        if simple_name in cleaned_names:
            return cleaned_names[simple_name]

    return None


def clean_points(points: pd.DataFrame, curve_label: str, alpha: float) -> pd.DataFrame:
    """Clean one digitized curve and apply the x/c shift correction."""
    cleaned = points.copy()
    cleaned["x"] = pd.to_numeric(cleaned["x"], errors="coerce")
    cleaned["cp"] = pd.to_numeric(cleaned["cp"], errors="coerce")
    cleaned = cleaned.dropna(subset=["x", "cp"])

    if len(cleaned) < 2:
        raise ValueError(f"Curve '{curve_label}' must contain at least two points.")

    # WebPlotDigitizer reads the shifted x_plot value from Figure 4.
    cleaned["x"] = cleaned["x"] - 0.05 * alpha

    # If the same x/c is clicked twice, use the average Cp value.
    cleaned = cleaned.groupby("x", as_index=False)["cp"].mean()
    cleaned = cleaned.sort_values("x").reset_index(drop=True)

    if len(cleaned) < 2:
        raise ValueError(f"Curve '{curve_label}' has fewer than two unique x values.")

    return cleaned


def load_curves_from_csv(csv_path: Path, alpha: float) -> list[pd.DataFrame]:
    """Load one digitized Cp curve from a CSV file."""
    frame = read_digitized_csv(csv_path)
    frame = frame.dropna(how="all").dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError(f"Digitized CSV file is empty: {csv_path}")

    dataset_column = find_column(frame, ["dataset", "curve", "series", "name"])
    x_column = find_column(frame, ["x", "x/c", "xoverc", "xcoordinate"])
    cp_column = find_column(frame, ["cp", "c_p", "pressurecoefficient", "y"])

    curves = []

    if dataset_column is not None and x_column is not None and cp_column is not None:
        for dataset_name, group in frame.groupby(dataset_column):
            points = pd.DataFrame({"x": group[x_column], "cp": group[cp_column]})
            label = f"{csv_path.stem}_{dataset_name}"
            curves.append(clean_points(points, label, alpha))
        return curves

    numeric_frame = pd.DataFrame()
    for column in frame.columns:
        numeric_frame[column] = pd.to_numeric(frame[column], errors="coerce")

    numeric_columns = []
    for column in numeric_frame.columns:
        if numeric_frame[column].notna().sum() >= 2:
            numeric_columns.append(column)

    if len(numeric_columns) < 2:
        raise ValueError(
            f"Could not find usable x/Cp columns in WebPlotDigitizer file: {csv_path}"
        )

    if len(numeric_columns) == 2:
        points = pd.DataFrame(
            {
                "x": numeric_frame[numeric_columns[0]],
                "cp": numeric_frame[numeric_columns[1]],
            }
        )
        curves.append(clean_points(points, csv_path.stem, alpha))
        return curves

    for column_index in range(0, len(numeric_columns) - 1, 2):
        x_name = numeric_columns[column_index]
        cp_name = numeric_columns[column_index + 1]
        points = pd.DataFrame({"x": numeric_frame[x_name], "cp": numeric_frame[cp_name]})
        label = f"{csv_path.stem}_pair_{(column_index // 2) + 1}"
        curves.append(clean_points(points, label, alpha))

    return curves


def load_x_grid(x_grid_file: Path) -> np.ndarray:
    """Read the shared x/c grid used by both datasets."""
    if not x_grid_file.exists():
        raise FileNotFoundError(
            f"Shared x-grid file was not found: {x_grid_file}. "
            "Create data/x_grid.csv before running the preprocessing scripts."
        )

    frame = pd.read_csv(x_grid_file)
    if "x" not in frame.columns:
        raise ValueError(f"X-grid file must contain an 'x' column: {x_grid_file}")
    x_values = pd.to_numeric(frame["x"], errors="raise").to_numpy(dtype=float)

    if len(x_values) < 2:
        raise ValueError("The shared x-grid must contain at least two points.")
    if not np.all(np.isfinite(x_values)):
        raise ValueError("The shared x-grid contains invalid numeric values.")

    return x_values


def find_common_x_range(digitized_curves: list[pd.DataFrame]) -> tuple[float, float]:
    """Find the x/c range covered by all digitized curves."""
    if not digitized_curves:
        raise ValueError("No digitized curves were loaded.")

    lower_limits = []
    upper_limits = []

    for curve in digitized_curves:
        lower_limits.append(float(curve["x"].min()))
        upper_limits.append(float(curve["x"].max()))

    common_min = max(lower_limits)
    common_max = min(upper_limits)

    if common_min >= common_max:
        raise ValueError(
            "The digitized alpha curves do not share an overlapping x/c range. "
            "Check the digitized CSV files and the alpha values in their filenames."
        )

    return common_min, common_max


def check_x_grid_inside_range(
    x_values: np.ndarray,
    common_min: float,
    common_max: float,
) -> None:
    """Make sure the shared grid does not require extrapolation."""
    tolerance = 1e-12
    for x_value in x_values:
        if x_value < common_min - tolerance or x_value > common_max + tolerance:
            raise ValueError(
                "data/x_grid.csv contains x/c values outside the digitized "
                "Gregory curve range. Do not extrapolate reference Cp values."
            )


def interpolate_to_target_grid(
    digitized_curve: pd.DataFrame,
    target_x_values: np.ndarray,
    curve_label: str,
) -> np.ndarray:
    """Interpolate one digitized curve onto the shared x/c grid."""
    source_x = digitized_curve["x"].to_numpy(dtype=float)
    source_cp = digitized_curve["cp"].to_numpy(dtype=float)

    smallest_x = source_x[0]
    largest_x = source_x[-1]
    target_min = float(np.min(target_x_values))
    target_max = float(np.max(target_x_values))

    if target_min < smallest_x or target_max > largest_x:
        raise ValueError(
            f"Curve '{curve_label}' covers x/c from {smallest_x:.5f} to "
            f"{largest_x:.5f}, but the target grid runs from {target_min:.5f} "
            f"to {target_max:.5f}. Digitize the full curve range or use a "
            "matching x-grid."
        )

    return np.interp(target_x_values, source_x, source_cp)


def interpolate_between_alpha_curves(
    alpha_values: list[float],
    cp_curves_on_grid: list[np.ndarray],
    target_x_values: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create 50 reference curves by interpolating between alpha values."""
    if len(alpha_values) < 2:
        raise ValueError(
            "At least two digitized alpha curves are needed to create "
            "50 interpolated reference curves."
        )

    sorted_pairs = sorted(zip(alpha_values, cp_curves_on_grid), key=lambda item: item[0])
    sorted_alpha_values = [pair[0] for pair in sorted_pairs]
    sorted_cp_curves = [pair[1] for pair in sorted_pairs]

    for index in range(1, len(sorted_alpha_values)):
        if sorted_alpha_values[index] == sorted_alpha_values[index - 1]:
            raise ValueError(
                f"Duplicate alpha value found: {sorted_alpha_values[index]}. "
                "Each digitized alpha curve must have a unique filename alpha."
            )

    alpha_array = np.array(sorted_alpha_values, dtype=float)
    cp_matrix = np.vstack(sorted_cp_curves)
    reference_alpha_values = np.linspace(
        alpha_array[0],
        alpha_array[-1],
        NUMBER_OF_REFERENCE_CURVES,
    )

    output_data = {"x": target_x_values}

    for reference_number, reference_alpha in enumerate(reference_alpha_values, start=1):
        interpolated_cp = []
        for point_index in range(len(target_x_values)):
            cp_at_same_x = cp_matrix[:, point_index]
            cp_value = np.interp(reference_alpha, alpha_array, cp_at_same_x)
            interpolated_cp.append(cp_value)

        output_data[f"reference_{reference_number}"] = np.array(interpolated_cp)

    return output_data


def prepare_reference_curves(
    digitized_directory: Path,
    output_file: Path,
    x_grid_file: Path,
) -> None:
    """Create data/reference_curves.csv from the Gregory source curves."""
    if not SOURCE_PDF.exists():
        print(f"Note: expected source PDF was not found at {SOURCE_PDF}")

    csv_files = find_digitized_csv_files(digitized_directory)

    alpha_values = []
    digitized_curves = []

    for csv_file in csv_files:
        alpha = extract_alpha_from_filename(csv_file)
        curves_from_file = load_curves_from_csv(csv_file, alpha)

        if len(curves_from_file) != 1:
            raise ValueError(
                f"File '{csv_file.name}' contains {len(curves_from_file)} curves. "
                "Use one digitized Cp curve per alpha file."
            )

        alpha_values.append(alpha)
        digitized_curves.append(curves_from_file[0])

    common_min, common_max = find_common_x_range(digitized_curves)

    target_x_values = load_x_grid(x_grid_file)
    check_x_grid_inside_range(target_x_values, common_min, common_max)

    cp_curves_on_grid = []

    for index, curve in enumerate(digitized_curves):
        alpha = alpha_values[index]
        curve_label = f"alpha_{alpha:g}"
        interpolated_cp = interpolate_to_target_grid(
            curve,
            target_x_values,
            curve_label,
        )
        cp_curves_on_grid.append(interpolated_cp)

    output_data = interpolate_between_alpha_curves(
        alpha_values,
        cp_curves_on_grid,
        target_x_values,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_data).to_csv(output_file, index=False)

    print(f"Created {output_file}")
    print(f"Digitized alpha curves loaded: {len(alpha_values)}")
    print(f"Alpha range: {min(alpha_values):g} to {max(alpha_values):g}")
    print(f"Common x/c range used: {common_min:.6g} to {common_max:.6g}")
    print(f"Number of reference curves: {NUMBER_OF_REFERENCE_CURVES}")
    print(f"Number of x/c points per curve: {len(target_x_values)}")


def main() -> None:
    """Run the Gregory reference-data preparation script."""
    parser = argparse.ArgumentParser(
        description="Prepare WebPlotDigitizer Cp exports as reference_curves.csv."
    )
    parser.add_argument(
        "--digitized-dir",
        default=str(SOURCE_CURVES_DIRECTORY),
        help="Folder containing WebPlotDigitizer CSV exports.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Output CSV file in the existing project format.",
    )
    parser.add_argument(
        "--x-grid-file",
        default=str(DEFAULT_X_GRID_FILE),
        help="CSV file containing the shared x/c grid.",
    )
    args = parser.parse_args()

    prepare_reference_curves(
        digitized_directory=Path(args.digitized_dir),
        output_file=Path(args.output),
        x_grid_file=Path(args.x_grid_file),
    )


if __name__ == "__main__":
    main()

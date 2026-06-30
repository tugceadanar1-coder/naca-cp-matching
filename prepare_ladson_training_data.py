"""Prepare Ladson experimental Cp data as training curves.

This script reads:

    data/raw/training_data/CP_Ladson.dat

and writes:

    data/training_curves.csv

The output format matches the main project:

    x, training_1, training_2, training_3, training_4

The script uses the shared x/c grid from data/x_grid.csv. The same grid is
also used by the Gregory reference-data script, so the training and reference
curves line up before SSE is calculated.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"
RAW_DIRECTORY = DATA_DIRECTORY / "raw"
LADSON_FILE = RAW_DIRECTORY / "training_data" / "CP_Ladson.dat"
X_GRID_FILE = DATA_DIRECTORY / "x_grid.csv"
OUTPUT_FILE = DATA_DIRECTORY / "training_curves.csv"


def parse_zone_header(line: str) -> dict[str, float | str]:
    """Read Reynolds number, alpha, and transition type from one zone line."""
    pattern = r'Re=(\d+(?:\.\d+)?) million, alpha=([-.]?\d+(?:\.\d+)?), (.*?)"'
    match = re.search(pattern, line)
    if match is None:
        raise ValueError(f"Could not parse Ladson zone header: {line}")

    return {
        "title": line.strip(),
        "re_million": float(match.group(1)),
        "alpha": float(match.group(2)),
        "transition": match.group(3).strip(),
    }


def read_ladson_zones(dat_path: Path) -> list[dict]:
    """Read all Cp zones from the Ladson .dat file."""
    if not dat_path.exists():
        raise FileNotFoundError(f"Ladson file was not found: {dat_path}")

    zones = []
    current_zone = None

    with dat_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue
            if line.startswith("#") or line.lower().startswith("variables"):
                continue

            if line.lower().startswith("zone"):
                if current_zone is not None:
                    zones.append(current_zone)

                current_zone = parse_zone_header(line)
                current_zone["points"] = []
                continue

            if current_zone is None:
                continue

            pieces = line.split()
            if len(pieces) >= 2:
                try:
                    x_value = float(pieces[0])
                    cp_value = float(pieces[1])
                    current_zone["points"].append((x_value, cp_value))
                except ValueError:
                    pass

    if current_zone is not None:
        zones.append(current_zone)

    if not zones:
        raise ValueError(f"No Ladson zones were found in: {dat_path}")

    return zones


def extract_upper_surface(points: list[tuple[float, float]]) -> pd.DataFrame:
    """Extract the upper-surface branch from one Ladson zone."""
    if len(points) < 2:
        raise ValueError("A Ladson zone must contain at least two points.")

    # Ladson zones contain two surface branches. The second branch starts at
    # the leading edge, x/c = 0, and runs toward the trailing edge. For positive
    # alpha this is the upper-surface suction side.
    leading_edge_index = None
    for index, point in enumerate(points):
        if abs(point[0]) < 1e-12:
            leading_edge_index = index

    if leading_edge_index is None:
        selected_points = points
    else:
        selected_points = points[leading_edge_index:]

    frame = pd.DataFrame(selected_points, columns=["x", "cp"])
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame["cp"] = pd.to_numeric(frame["cp"], errors="coerce")
    frame = frame.dropna(subset=["x", "cp"])

    # Average duplicate x/c points, then sort from leading edge to trailing edge.
    frame = frame.groupby("x", as_index=False)["cp"].mean()
    frame = frame.sort_values("x").reset_index(drop=True)

    if len(frame) < 2:
        raise ValueError("Upper-surface branch has fewer than two usable x/c points.")

    return frame


def add_upper_surface_data(zones: list[dict]) -> None:
    """Attach cleaned upper-surface data to each parsed zone."""
    for zone in zones:
        zone["upper_surface"] = extract_upper_surface(zone["points"])


def select_four_training_zones(zones: list[dict]) -> list[dict]:
    """Select four representative Ladson Cp curves."""
    if len(zones) < 4:
        raise ValueError("At least four Ladson zones are needed.")

    selected = []

    # Use the Re=6 million free-transition alpha sweep first. It gives a clear
    # low-, medium-, and high-alpha set from the same experimental condition.
    for zone in zones:
        same_re = abs(float(zone["re_million"]) - 6.0) < 1e-9
        is_free = "free" in str(zone["transition"]).lower()
        if same_re and is_free:
            selected.append(zone)

    selected = sorted(selected, key=lambda zone: float(zone["alpha"]))

    if len(selected) > 3:
        selected = selected[:3]

    # Add one high-Re fixed-transition high-alpha case as the fourth curve.
    remaining = []
    for zone in zones:
        if zone not in selected:
            remaining.append(zone)

    high_alpha_limit = max(float(zone["alpha"]) for zone in remaining) - 0.2
    high_alpha_candidates = []
    for zone in remaining:
        if float(zone["alpha"]) >= high_alpha_limit:
            high_alpha_candidates.append(zone)

    if high_alpha_candidates:
        fourth_zone = max(
            high_alpha_candidates,
            key=lambda zone: float(zone["re_million"]),
        )
        selected.append(fourth_zone)

    # Fallback: if the file structure changes, fill missing slots by alpha order.
    if len(selected) < 4:
        for zone in sorted(zones, key=lambda zone: float(zone["alpha"])):
            if zone not in selected:
                selected.append(zone)
            if len(selected) == 4:
                break

    if len(selected) != 4:
        raise ValueError("Could not select exactly four Ladson training curves.")

    return selected


def load_x_grid(x_grid_file: Path) -> np.ndarray:
    """Read the shared x/c grid from data/x_grid.csv."""
    if not x_grid_file.exists():
        raise FileNotFoundError(f"Shared x-grid file was not found: {x_grid_file}")

    frame = pd.read_csv(x_grid_file)
    if "x" not in frame.columns:
        raise ValueError("data/x_grid.csv must contain an 'x' column.")

    x_values = pd.to_numeric(frame["x"], errors="raise").to_numpy(dtype=float)
    if len(x_values) < 2:
        raise ValueError("data/x_grid.csv must contain at least two x/c values.")
    if not np.all(np.isfinite(x_values)):
        raise ValueError("data/x_grid.csv contains invalid x/c values.")

    return x_values


def check_x_grid_inside_ladson_range(
    x_grid_values: np.ndarray,
    selected_zones: list[dict],
) -> tuple[float, float]:
    """Check that the shared x/c grid fits inside all selected Ladson curves."""
    lower_limits = []
    upper_limits = []

    for zone in selected_zones:
        curve = zone["upper_surface"]
        lower_limits.append(float(curve["x"].min()))
        upper_limits.append(float(curve["x"].max()))

    common_min = max(lower_limits)
    common_max = min(upper_limits)

    if common_min >= common_max:
        raise ValueError("Selected Ladson curves do not share an x/c overlap.")

    tolerance = 1e-12
    grid_min = float(np.min(x_grid_values))
    grid_max = float(np.max(x_grid_values))

    if grid_min < common_min - tolerance or grid_max > common_max + tolerance:
        raise ValueError(
            "data/x_grid.csv contains x/c values outside the selected Ladson "
            "curve range. Do not extrapolate training Cp values."
        )

    return common_min, common_max


def interpolate_training_curves(
    selected_zones: list[dict],
    target_x_values: np.ndarray,
) -> pd.DataFrame:
    """Interpolate selected Ladson curves onto the target x/c grid."""
    output_data = {"x": target_x_values}

    for training_number, zone in enumerate(selected_zones, start=1):
        curve = zone["upper_surface"]
        source_x = curve["x"].to_numpy(dtype=float)
        source_cp = curve["cp"].to_numpy(dtype=float)

        target_min = float(target_x_values.min())
        target_max = float(target_x_values.max())
        if target_min < source_x.min() or target_max > source_x.max():
            raise ValueError(
                f"training_{training_number} would require extrapolation. "
                "The script only interpolates inside measured x/c ranges."
            )

        output_data[f"training_{training_number}"] = np.interp(
            target_x_values,
            source_x,
            source_cp,
        )

    return pd.DataFrame(output_data)


def describe_zone(zone: dict) -> str:
    """Create a short readable description of one selected Ladson zone."""
    return (
        f"Re={float(zone['re_million']):g} million, "
        f"alpha={float(zone['alpha']):g}, "
        f"{zone['transition']}"
    )


def prepare_ladson_training_data() -> None:
    """Create data/training_curves.csv from CP_Ladson.dat."""
    zones = read_ladson_zones(LADSON_FILE)
    add_upper_surface_data(zones)

    selected_zones = select_four_training_zones(zones)
    target_x_values = load_x_grid(X_GRID_FILE)
    common_min, common_max = check_x_grid_inside_ladson_range(
        target_x_values,
        selected_zones,
    )

    training_table = interpolate_training_curves(selected_zones, target_x_values)
    training_table.to_csv(OUTPUT_FILE, index=False)

    print(f"Read Ladson file: {LADSON_FILE}")
    print("Selected Ladson curves:")
    for index, zone in enumerate(selected_zones, start=1):
        print(f"training_{index}: {describe_zone(zone)}")
    print(f"Common Ladson x/c range: {common_min:.6g} to {common_max:.6g}")
    print(
        f"Written x/c range: {training_table['x'].min():.6g} "
        f"to {training_table['x'].max():.6g}"
    )
    print(f"Created {OUTPUT_FILE}")
    print(f"Rows: {len(training_table)}")
    print(f"Columns: {len(training_table.columns)}")
    print("Output columns: " + ", ".join(training_table.columns))


if __name__ == "__main__":
    prepare_ladson_training_data()

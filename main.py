"""Start the NACA Cp matching program."""

from __future__ import annotations

from src.app import AssignmentApp


def main() -> None:
    """Run the project."""
    try:
        output_paths = AssignmentApp().run()
    except Exception as exc:
        raise SystemExit(f"Cp matching failed: {exc}") from exc

    print("Cp matching completed successfully.")
    for output_name, output_path in output_paths.items():
        print(f"- {output_name}: {output_path}")


if __name__ == "__main__":
    main()

# Least-Squares Based Aerodynamic Pressure Curve Matching

## Project overview

This project compares NACA 0012 pressure coefficient (`Cp`) curves using a
least-squares matching method. The goal is to identify which reference Cp curve
is closest to each training Cp curve on the same `x/c` grid.

The project is an explainable data-analysis workflow. It does not use machine
learning, does not predict new aerodynamic data, and does not perform CFD
simulation. It only compares existing experimental Cp curves.

## Experimental data sources

The project uses two independent experimental data sources:

- Gregory & O'Reilly (1970), Figure 4
  - digitized with WebPlotDigitizer
  - stored in `data/raw/source_curves/`
  - used to generate `data/reference_curves.csv`

- Ladson pressure data
  - stored in `data/raw/training_data/CP_Ladson.dat`
  - used to generate `data/training_curves.csv`

The processed reference and training curves use the same shared `x/c` grid:

```text
x/c = 0.01 to 0.94
```

The grid is stored in:

```text
data/x_grid.csv
```

Both preprocessing scripts read this same file. This is important because SSE
can only be calculated correctly when the training and reference Cp values are
compared at the same `x/c` positions. The scripts do not extrapolate Cp values
outside the measured or digitized data range.

## Mathematical method

For each training curve and each reference curve, the code calculates the
point-by-point Cp difference:

```text
Delta Cp_i = Cp_training(x_i) - Cp_reference(x_i)
```

The main matching metric is the sum of squared errors:

```text
SSE = sum_i (Delta Cp_i)^2
```

The best reference curve is the one with the smallest SSE:

```text
best_reference = argmin_j SSE_j
```

The project also calculates:

```text
RMSE = sqrt(SSE / n)
max |Delta Cp| = max_i |Cp_training(x_i) - Cp_reference(x_i)|
```

The Top-3 closest reference curves are stored for each training curve. A
confidence margin is also reported:

```text
confidence_margin = SSE_rank2 - SSE_rank1
relative_confidence_margin = (SSE_rank2 - SSE_rank1) / SSE_rank1
```

The confidence margin does not change the best match. It only indicates whether
the best match is clearly separated from the second-best candidate.

## Main files

Raw data:

- `data/raw/gregory_oreilly_1970.pdf`
- `data/raw/source_curves/cp_alpha_*.csv`
- `data/raw/training_data/CP_Ladson.dat`

Processed data:

- `data/x_grid.csv`
- `data/reference_curves.csv`
- `data/training_curves.csv`
- `data/test_points.csv`

Preprocessing scripts:

- `prepare_digitized_cp_data.py`
- `prepare_ladson_training_data.py`

Main workflow:

- `main.py`

Source code:

- `src/data_loader.py`
- `src/curves.py`
- `src/matcher.py`
- `src/mapper.py`
- `src/database.py`
- `src/visualizer.py`
- `src/app.py`

## Running the project

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Regenerate the processed data in this order:

1. Check that `data/x_grid.csv` contains the shared `x/c` grid.
2. Prepare the Gregory & O'Reilly reference curves.
3. Prepare the Ladson training curves.

Prepare the reference curves:

```bash
python prepare_digitized_cp_data.py
```

Prepare the training curves:

```bash
python prepare_ladson_training_data.py
```

After these two scripts finish, `data/reference_curves.csv` and
`data/training_curves.csv` should have exactly the same `x` column.

Run the matching workflow:

```bash
python main.py
```

Run the unit tests:

```bash
python -m unittest discover tests
```

## Results

The workflow writes:

- `results/error_audit.csv`
- `results/best_matches.csv`
- `results/mapping_results.csv`
- `results/naca_cp_matching.sqlite`
- `results/cp_matching_plot.html`

`error_audit.csv` contains every training-reference comparison.  
`best_matches.csv` contains the selected best reference for each training curve.  
`mapping_results.csv` checks real test Cp points against the selected best
reference curve using the acceptance rule in the code.  
`naca_cp_matching.sqlite` stores the project tables in SQLite format.  
`cp_matching_plot.html` gives an interactive Bokeh visualization.

## Archived sample-data script

The old artificial sample-data generator has been moved to:

```text
archive/create_sample_data_archived.py
```

It is kept only as historical test material and is not part of the current
real experimental data workflow.

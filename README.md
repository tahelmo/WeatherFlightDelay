# WeatherFlightDelay

End-to-end data prep and baseline modeling for flight delay prediction using USDOT flight data and (optionally) NOAA GSOD weather.

## Setup
- Use Python 3.10+.
- Create a virtual environment and install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

## Data preparation
`src/prepare_data.py` streams the raw flights table, builds the `delay_15` label, adds airport/airline metadata, optionally merges weather, and writes a modeling-ready CSV.

Example (default paths assume data is under `data/raw`):
```bash
python src/prepare_data.py
```

## Baseline models
`models/baselines.py` trains Logistic Regression, Decision Tree, Random Forest, and an MLP on `data/processed/flights_prepared.csv` (expects `FL_DATE` has been dropped in that file).

Run from the repo root:
```bash
python models/baselines.py
```
It prints metrics per model, shows confusion matrices, and writes `baselines_metrics_summary.csv` in the `models` folder.

## Notebooks
Which we used in google colab
- `models/SMOTE_training.ipynb`
- `models/hyperparameter_tuning_experience.ipynb`


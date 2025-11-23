"""
Build a model-ready dataset: impute missing values, one-hot encode categoricals,
and scale numeric features, with chronological train/val/test splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split the dataframe into train/val/test by FL_DATE order."""
    df = df.sort_values("FL_DATE")
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    return train_df, val_df, test_df


def build_preprocessor(categorical_cols, numeric_cols) -> ColumnTransformer:
    """Create a preprocessing ColumnTransformer."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["FL_DATE"])
    return df


def main():
    parser = argparse.ArgumentParser(description="Preprocess flight delay dataset.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/processed/flights_prepared.csv"),
        help="Input CSV from prepare_data.py",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/flights_model_ready.joblib"),
        help="Where to save transformed splits and preprocessor.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train proportion (chronological).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation proportion (chronological).",
    )
    args = parser.parse_args()

    df = load_dataset(args.input_path)

    categorical_cols = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "origin_state",
        "destination_state",
        "station_id",
    ]
    numeric_cols = [
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "SCHEDULED_DEPARTURE_MIN",
        "SCHEDULED_ARRIVAL_MIN",
        "DISTANCE",
        "origin_latitude",
        "origin_longitude",
        "destination_latitude",
        "destination_longitude",
        "temp_f",
        "dewpoint_f",
        "visibility_miles",
        "wind_speed_knots",
        "precip_in",
        "snow_depth_in",
        "rain_flag",
        "snow_flag",
        "thunder_flag",
    ]

    # Chronological split on the full dataframe to preserve FL_DATE ordering.
    train_df, val_df, test_df = chronological_split(
        df, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    y_train = train_df["delay_15"].astype("int8").to_numpy()
    y_val = val_df["delay_15"].astype("int8").to_numpy()
    y_test = test_df["delay_15"].astype("int8").to_numpy()

    feature_cols = categorical_cols + numeric_cols
    X_train_df = train_df[feature_cols].copy()
    X_val_df = val_df[feature_cols].copy()
    X_test_df = test_df[feature_cols].copy()

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    output = {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(output, args.output_path)
    print(f"Saved model-ready data to {args.output_path}")


if __name__ == "__main__":
    main()

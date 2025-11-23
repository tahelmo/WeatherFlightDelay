"""
Data preparation script for the flight delay project.

This script:
1. Cleans the USDOT flight table (drops cancelled/diverted flights, builds a ≥15 min delay label).
2. Adds airline names and airport location metadata.
3. Outputs a modeling-ready CSV in data/processed.

Notes:
- Defaults keep only schedule-time features to avoid leakage (no actual departure/arrival times).
- The script processes the large flights CSV in chunks so it can run on modest hardware.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def parse_hhmm(value) -> Optional[int]:
    """Convert times like 2359 → minutes since midnight. Return None if malformed."""
    if pd.isna(value):
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None

    hours, minutes = divmod(value_int, 100)
    if hours > 24 or minutes >= 60:
        return None
    return hours * 60 + minutes


def load_airlines(path: Path) -> pd.Series:
    """Load airline name lookup keyed by IATA code."""
    airlines = pd.read_csv(path, dtype={"IATA_CODE": "string", "AIRLINE": "string"})
    return airlines.set_index("IATA_CODE")["AIRLINE"]


def load_airports(path: Path) -> pd.DataFrame:
    """Load airport metadata keyed by IATA code."""
    cols = ["IATA_CODE", "CITY", "STATE", "COUNTRY", "LATITUDE", "LONGITUDE"]
    airports = pd.read_csv(path, usecols=cols, dtype={"IATA_CODE": "string"})
    airports = airports.rename(
        columns={
            "CITY": "city",
            "STATE": "state",
            "COUNTRY": "country",
            "LATITUDE": "latitude",
            "LONGITUDE": "longitude",
        }
    )
    return airports.set_index("IATA_CODE")


def clean_flights_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Clean one chunk of the flights table and engineer core features."""
    # Drop flights we cannot model (cancelled/diverted) and missing delays.
    chunk = chunk[(chunk["CANCELLED"] == 0) & (chunk["DIVERTED"] == 0)]
    chunk["ARRIVAL_DELAY"] = pd.to_numeric(chunk["ARRIVAL_DELAY"], errors="coerce")
    chunk = chunk[chunk["ARRIVAL_DELAY"].notna()]

    # Label: arrival delay of 15+ minutes.
    chunk["delay_15"] = (chunk["ARRIVAL_DELAY"] >= 15).astype("int8")

    # Time features at schedule time only (avoid leakage).
    chunk["SCHEDULED_DEPARTURE_MIN"] = chunk["SCHEDULED_DEPARTURE"].apply(parse_hhmm)
    chunk["SCHEDULED_ARRIVAL_MIN"] = chunk["SCHEDULED_ARRIVAL"].apply(parse_hhmm)
    chunk["DISTANCE"] = pd.to_numeric(chunk["DISTANCE"], errors="coerce")

    chunk = chunk.dropna(
        subset=["SCHEDULED_DEPARTURE_MIN", "SCHEDULED_ARRIVAL_MIN", "DISTANCE"]
    )

    # Flight date for chronological splits.
    chunk["FL_DATE"] = pd.to_datetime(
        dict(year=chunk["YEAR"], month=chunk["MONTH"], day=chunk["DAY"])
    )

    feature_cols = [
        "FL_DATE",
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE_MIN",
        "SCHEDULED_ARRIVAL_MIN",
        "DISTANCE",
        "delay_15",
    ]
    return chunk[feature_cols]


def enrich_with_metadata(
    flights: pd.DataFrame, airlines: pd.Series, airports: pd.DataFrame
) -> pd.DataFrame:
    """Add airline names and airport coordinates."""
    flights["airline_name"] = flights["AIRLINE"].map(airlines)

    # Airport lookups.
    flights["origin_latitude"] = flights["ORIGIN_AIRPORT"].map(
        airports["latitude"]
    )
    flights["origin_longitude"] = flights["ORIGIN_AIRPORT"].map(
        airports["longitude"]
    )
    flights["destination_latitude"] = flights["DESTINATION_AIRPORT"].map(
        airports["latitude"]
    )
    flights["destination_longitude"] = flights["DESTINATION_AIRPORT"].map(
        airports["longitude"]
    )

    flights["origin_state"] = flights["ORIGIN_AIRPORT"].map(airports["state"])
    flights["destination_state"] = flights["DESTINATION_AIRPORT"].map(
        airports["state"]
    )

    return flights


def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance (km) between coordinate arrays."""
    r = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r * c


def load_station_metadata(path: Path, year: int) -> pd.DataFrame:
    """Load NOAA station metadata (ISD history) for a specific year."""
    stations = pd.read_csv(
        path,
        dtype={"USAF": "string", "WBAN": "string", "STATE": "string", "CTRY": "string"},
    )
    stations = stations.rename(columns={"LAT": "latitude", "LON": "longitude"})
    stations["BEGIN"] = pd.to_numeric(stations["BEGIN"], errors="coerce")
    stations["END"] = pd.to_numeric(stations["END"], errors="coerce")
    stations = stations[
        (stations["CTRY"] == "US")
        & stations["latitude"].notna()
        & stations["longitude"].notna()
        & (stations["BEGIN"] <= int(f"{year}0101"))
        & (stations["END"] >= int(f"{year}1231"))
    ]
    stations["station_id"] = stations["USAF"].str.zfill(6) + "-" + stations["WBAN"].str.zfill(5)
    stations = stations.set_index("station_id")
    return stations


def map_airports_to_stations(
    airports: pd.DataFrame, stations: pd.DataFrame
) -> pd.DataFrame:
    """Assign each airport to the nearest NOAA station."""
    airport_codes: List[str] = []
    station_ids: List[str] = []

    station_coords = stations[["latitude", "longitude"]].to_numpy(dtype=float)
    station_index = stations.index.to_list()

    for code, row in airports.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            continue
        dists = haversine_distance(lat, lon, station_coords[:, 0], station_coords[:, 1])
        nearest_idx = int(np.argmin(dists))
        airport_codes.append(code)
        station_ids.append(station_index[nearest_idx])

    mapping = pd.DataFrame({"airport": airport_codes, "station_id": station_ids})
    return mapping


def parse_prcp(value) -> Optional[float]:
    """Parse precipitation field that may include trace flags (e.g., '0.12G')."""
    if pd.isna(value):
        return None
    text = str(value)
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    if not cleaned:
        return None
    val = float(cleaned)
    if val >= 99.0:  # NOAA missing sentinel (99.99)
        return None
    return val


def parse_numeric_with_flag(value) -> Optional[float]:
    """Strip trailing flags (e.g., '19.9*') and return float; treat 999.9 sentinel as missing."""
    if pd.isna(value):
        return None
    text = str(value).rstrip("*ABCDEFGHIJKLmnopqrstuvwxyz")
    try:
        val = float(text)
    except ValueError:
        return None
    if val >= 999.0:
        return None
    return val


def load_gsod_year(
    gsod_dir: Path, year: int, stations_filter: Optional[set[str]] = None
) -> pd.DataFrame:
    """Load GSOD daily weather for a year; optionally filter to station ids."""
    cols = [
        "STN",
        "WBAN",
        "YEARMODA",
        "TEMP",
        "TEMP_CNT",
        "DEWP",
        "DEWP_CNT",
        "SLP",
        "SLP_CNT",
        "STP",
        "STP_CNT",
        "VISIB",
        "VISIB_CNT",
        "WDSP",
        "WDSP_CNT",
        "MXSPD",
        "GUST",
        "MAX",
        "MIN",
        "PRCP",
        "SNDP",
        "FRSHTT",
    ]
    colspecs = [
        (0, 6),    # STN
        (7, 12),   # WBAN
        (14, 22),  # YEARMODA
        (24, 30),  # TEMP
        (32, 33),  # TEMP_CNT
        (35, 41),  # DEWP
        (43, 44),  # DEWP_CNT
        (46, 52),  # SLP
        (54, 55),  # SLP_CNT
        (57, 63),  # STP
        (65, 66),  # STP_CNT
        (68, 73),  # VISIB
        (75, 76),  # VISIB_CNT
        (78, 83),  # WDSP
        (85, 86),  # WDSP_CNT
        (88, 93),  # MXSPD
        (95, 100), # GUST
        (102, 108),# MAX
        (110, 116),# MIN
        (118, 123),# PRCP
        (124, 130),# SNDP
        (131, 138),# FRSHTT
    ]
    records: List[pd.DataFrame] = []
    files = list(gsod_dir.glob("*.op.gz"))
    if not files:
        raise RuntimeError(f"No GSOD .op.gz files found in {gsod_dir}")
    for path in files:
        station_id = "-".join(path.stem.split("-")[:2])
        if stations_filter and station_id not in stations_filter:
            continue
        df = pd.read_fwf(
            path,
            colspecs=colspecs,
            skiprows=1,
            names=cols,
            compression="gzip",
            dtype=str,
        )
        if df.empty:
            continue
        df["station_id"] = station_id
        records.append(df)

    if not records:
        raise RuntimeError(
            "No GSOD files loaded; check station filter—"
            f" {len(files)} files present, {len(stations_filter or [])} stations in filter."
        )

    weather = pd.concat(records, ignore_index=True)
    weather["date"] = pd.to_datetime(weather["YEARMODA"], format="%Y%m%d", errors="coerce")
    weather = weather[weather["date"].dt.year == year]
    weather["station_id"] = weather["station_id"].astype(str)

    weather["temp_f"] = weather["TEMP"].astype(float)
    weather["dewpoint_f"] = weather["DEWP"].astype(float)
    weather["visibility_miles"] = pd.to_numeric(weather["VISIB"], errors="coerce")
    weather["wind_speed_knots"] = pd.to_numeric(weather["WDSP"], errors="coerce")
    weather["precip_in"] = weather["PRCP"].apply(parse_prcp)
    weather["snow_depth_in"] = weather["SNDP"].apply(parse_numeric_with_flag)

    weather["rain_flag"] = weather["FRSHTT"].fillna("").str[1:2].eq("1").astype("int8")
    weather["snow_flag"] = weather["FRSHTT"].fillna("").str[2:3].eq("1").astype("int8")
    weather["thunder_flag"] = weather["FRSHTT"].fillna("").str[4:5].eq("1").astype("int8")

    feature_cols = [
        "station_id",
        "date",
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
    return weather[feature_cols]


def attach_weather(
    flights: pd.DataFrame,
    airports: pd.DataFrame,
    noaa_stations_path: Path,
    gsod_dir: Path,
    year: int = 2015,
) -> pd.DataFrame:
    """Merge origin-airport weather onto flights (left join to avoid dropping flights)."""
    stations = load_station_metadata(noaa_stations_path, year)
    mapping = map_airports_to_stations(airports, stations)

    station_ids = set(mapping["station_id"].unique())
    if not station_ids:
        raise RuntimeError("No station_ids mapped to airports; check station metadata filters.")

    weather = load_gsod_year(gsod_dir, year, stations_filter=station_ids)
    weather["_wx_date_str"] = pd.to_datetime(weather["date"]).dt.strftime("%Y-%m-%d")

    # Attach station_id to flights, then merge on station_id + date.
    flights = flights.copy()
    flights = flights.merge(mapping, left_on="ORIGIN_AIRPORT", right_on="airport", how="left")
    flights["_fl_date_str"] = pd.to_datetime(flights["FL_DATE"]).dt.strftime("%Y-%m-%d")

    merged = flights.merge(
        weather,
        left_on=["station_id", "_fl_date_str"],
        right_on=["station_id", "_wx_date_str"],
        how="left",
    )
    merged = merged.drop(columns=["airport", "date", "_fl_date_str", "_wx_date_str"])
    return merged


def process_flights(
    flights_path: Path,
    airlines_path: Path,
    airports_path: Path,
    output_path: Path,
    chunksize: int = 200_000,
    max_rows: Optional[int] = 500_000,
    include_weather: bool = True,
    noaa_stations_path: Optional[Path] = None,
    gsod_dir: Optional[Path] = None,
    year: int = 2015,
) -> Path:
    """Stream the flights CSV, clean/enrich, and write the processed dataset."""
    airlines = load_airlines(airlines_path)
    airports = load_airports(airports_path)

    processed_chunks: List[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(
        flights_path,
        chunksize=chunksize,
        dtype={"AIRLINE": "string", "ORIGIN_AIRPORT": "string", "DESTINATION_AIRPORT": "string"},
    ):
        clean_chunk = clean_flights_chunk(chunk)
        clean_chunk = enrich_with_metadata(clean_chunk, airlines, airports)

        if max_rows is not None and total_rows + len(clean_chunk) > max_rows:
            remaining = max_rows - total_rows
            clean_chunk = clean_chunk.iloc[:remaining]

        processed_chunks.append(clean_chunk)
        total_rows += len(clean_chunk)

        if max_rows is not None and total_rows >= max_rows:
            break

    if not processed_chunks:
        raise RuntimeError("No rows processed; check filters and input paths.")

    df = pd.concat(processed_chunks, ignore_index=True)
    df.sort_values(["FL_DATE", "SCHEDULED_DEPARTURE_MIN"], inplace=True)

    if include_weather:
        if noaa_stations_path is None or gsod_dir is None:
            raise ValueError("Weather integration requires noaa_stations_path and gsod_dir.")
        before = len(df)
        df = attach_weather(df, airports, noaa_stations_path, gsod_dir, year=year)
        if df.empty:
            raise RuntimeError("Weather merge produced zero rows; check station mapping/GSOD files.")
        print(f"Weather merge: {len(df)} rows (from {before})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare flight delay dataset.")
    parser.add_argument(
        "--flights-path",
        default=Path("data/raw/USDOT/flights.csv"),
        type=Path,
        help="Path to USDOT flights.csv",
    )
    parser.add_argument(
        "--airlines-path",
        default=Path("data/raw/USDOT/airlines.csv"),
        type=Path,
        help="Path to airlines.csv lookup",
    )
    parser.add_argument(
        "--airports-path",
        default=Path("data/raw/USDOT/airports.csv"),
        type=Path,
        help="Path to airports.csv lookup",
    )
    parser.add_argument(
        "--output-path",
        default=Path("data/processed/flights_prepared.csv"),
        type=Path,
        help="Where to write the processed dataset (CSV).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200_000,
        help="Rows per chunk when streaming the flights table.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Optional cap on processed rows to keep the dataset manageable.",
    )
    parser.add_argument(
        "--no-weather",
        dest="include_weather",
        action="store_false",
        help="Skip NOAA GSOD weather merge (default is to include weather).",
    )
    parser.set_defaults(include_weather=True)
    parser.add_argument(
        "--noaa-stations-path",
        type=Path,
        default=Path("data/raw/NOAA/isd-history.csv"),
        help="Path to NOAA ISD station metadata (isd-history.csv).",
    )
    parser.add_argument(
        "--gsod-dir",
        type=Path,
        default=Path("data/raw/NOAA/gsod_2015"),
        help="Directory containing GSOD .op.gz files for the year.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2015,
        help="Year of data to process (for station filtering and GSOD files).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = process_flights(
        flights_path=args.flights_path,
        airlines_path=args.airlines_path,
        airports_path=args.airports_path,
        output_path=args.output_path,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
        include_weather=args.include_weather,
        noaa_stations_path=args.noaa_stations_path,
        gsod_dir=args.gsod_dir,
        year=args.year,
    )
    print(f"Wrote processed dataset to {output_path}")


if __name__ == "__main__":
    main()

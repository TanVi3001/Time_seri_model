"""
Merge Investing.com CSV exports into the canonical dataset used by the
RNN/LSTM notebooks.

Run from this repository root:
    python merge_investing_data.py

Output:
    data/copper_investing_2006_2026.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw_investing"
OUTPUT_PATH = BASE_DIR / "data" / "copper_investing_2006_2026.csv"

START_DATE = pd.Timestamp("2006-01-01")
END_DATE = pd.Timestamp("2026-09-26")


def parse_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan", "None"}:
        return np.nan
    return pd.to_numeric(text, errors="coerce")


def parse_volume(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper().replace(",", "")
    if text in {"", "-", "NAN", "NONE"}:
        return np.nan

    multiplier = 1.0
    if text.endswith("K"):
        multiplier, text = 1_000.0, text[:-1]
    elif text.endswith("M"):
        multiplier, text = 1_000_000.0, text[:-1]
    elif text.endswith("B"):
        multiplier, text = 1_000_000_000.0, text[:-1]

    return pd.to_numeric(text, errors="coerce") * multiplier


def read_investing_csv(path, price_column, include_volume=False):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    raw = pd.read_csv(path)
    required = {"Date", "Price"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(
                raw["Date"], format="%m/%d/%Y", errors="coerce"
            ),
            price_column: raw["Price"].map(parse_number),
        }
    )

    if include_volume:
        if "Vol." not in raw.columns:
            raise ValueError(f"{path.name} has no Vol. column.")
        result["volume_copper"] = raw["Vol."].map(parse_volume)

    return result.dropna(subset=["date", price_column])


def merge_same_series(files, price_column, include_volume=False):
    frames = [
        read_investing_csv(
            RAW_DIR / filename,
            price_column=price_column,
            include_volume=include_volume,
        )
        for filename in files
    ]
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("date").drop_duplicates("date", keep="last")
    return result.reset_index(drop=True)


def build_dataset():
    series = [
        merge_same_series(
            ["Copper Futures Historical Data.csv", "copper_2016_2026.csv"],
            "closed_copper_price",
            include_volume=True,
        ),
        merge_same_series(
            ["Crude Oil WTI Futures Historical Data.csv", "wti_2016_2026.csv"],
            "close_wti_oil",
        ),
        merge_same_series(["Gold Futures Historical Data.csv"], "close_gold"),
        merge_same_series(["Silver Futures Historical Data.csv"], "close_silver"),
        merge_same_series(
            ["US Dollar Index Historical Data.csv", "dollar_index_2016_2026.csv"],
            "close_dollar_index",
        ),
        merge_same_series(
            [
                "United States 10-Year Bond Yield Historical Data.csv",
                "t_bill_2016_2026.csv",
            ],
            "close_t_bill",
        ),
        merge_same_series(
            ["NASDAQ Composite Historical Data.csv", "nasdaq_2016_2026.csv"],
            "close_Nasdaq",
        ),
        merge_same_series(
            [
                "Dow Jones Industrial Average Historical Data.csv",
                "dow_jones_2016_2026.csv",
            ],
            "close_Dow_Jones_Index",
        ),
    ]

    merged = series[0]
    for frame in series[1:]:
        merged = merged.merge(frame, on="date", how="inner")

    columns = [
        "date",
        "closed_copper_price",
        "volume_copper",
        "close_dollar_index",
        "close_t_bill",
        "close_Nasdaq",
        "close_Dow_Jones_Index",
        "close_wti_oil",
        "close_gold",
        "close_silver",
    ]

    merged = merged[columns]
    merged = merged.loc[
        (merged["date"] >= START_DATE) & (merged["date"] <= END_DATE)
    ]
    merged = merged.sort_values("date").drop_duplicates("date")
    value_columns = columns[1:]
    merged[value_columns] = merged[value_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    merged = merged.dropna(subset=value_columns).reset_index(drop=True)

    if merged.empty:
        raise ValueError("The merged dataset is empty.")
    if merged["date"].duplicated().any():
        raise ValueError("The merged dataset still has duplicate dates.")
    if merged[value_columns].isna().any().any():
        raise ValueError("The merged dataset still has missing values.")

    return merged


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(f"Dates: {dataset['date'].min().date()} -> {dataset['date'].max().date()}")
    print(f"Missing cells: {int(dataset.isna().sum().sum())}")
    print(f"Duplicate dates: {int(dataset['date'].duplicated().sum())}")


if __name__ == "__main__":
    main()

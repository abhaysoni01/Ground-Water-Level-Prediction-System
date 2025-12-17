import pandas as pd
import re

def load_and_preprocess(path):
    # Load CSV (Govt datasets are not UTF-8)
    df = pd.read_csv(path, encoding="latin1")

    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    # -----------------------------
    # Detect STATE column robustly
    # -----------------------------
    state_col = None
    for col in df.columns:
        if "state" in col:
            state_col = col
            break

    if state_col is None:
        raise ValueError("State column not found")

    df.rename(columns={state_col: "state"}, inplace=True)

    # -----------------------------
    # Identify groundwater columns
    # -----------------------------
    gw_cols = [
        col for col in df.columns
        if "meters below ground level" in col
    ]

    if not gw_cols:
        raise ValueError("No groundwater level columns found")

    # -----------------------------
    # Melt wide → long
    # -----------------------------
    df_long = df.melt(
        id_vars=["state"],
        value_vars=gw_cols,
        var_name="season_year",
        value_name="groundwater_level"
    )

    # Convert groundwater values safely
    df_long["groundwater_level"] = pd.to_numeric(
        df_long["groundwater_level"],
        errors="coerce"   # handles 'Dry'
    )

    df_long = df_long.dropna(subset=["groundwater_level"])

    # -----------------------------
    # Extract year & season
    # -----------------------------
    df_long["year"] = df_long["season_year"].apply(
        lambda x: int(re.search(r"(\d{4})", x).group(1))
    )

    df_long["season"] = df_long["season_year"].apply(
        lambda x: "pre-monsoon" if "pre" in x else "post-monsoon"
    )

    # Season-based timestamp
    df_long["month"] = df_long["season"].map({
        "pre-monsoon": 4,
        "post-monsoon": 10
    })

    df_long["timestamp"] = pd.to_datetime(
        df_long["year"].astype(str) + "-" +
        df_long["month"].astype(str) + "-01"
    )

    df_long["dayofyear"] = df_long["timestamp"].dt.dayofyear

    return df_long[
        ["state", "timestamp", "year", "month", "dayofyear", "groundwater_level"]
    ]

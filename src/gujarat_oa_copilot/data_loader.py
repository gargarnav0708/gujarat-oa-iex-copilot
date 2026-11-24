"""
Data Loading & Preprocessing Module for IEXInsiderMCP Market Records.
Parses 15-minute blocks across DAM, RTM, and GDAM markets and synchronizes them
into a unified time-indexed feature table.
"""

import os
import pandas as pd


DEFAULT_CSV_PATHS = [
    os.path.join(os.getcwd(), "IEXInsiderMCP", "IEX_Market_Data.csv"),
    os.path.join(os.getcwd(), "data", "IEX_Market_Data.csv"),
    "/Users/herr_ag98/.gemini/antigravity/scratch/gujarat-oa-iex-copilot/IEXInsiderMCP/IEX_Market_Data.csv",
]


def resolve_data_path(csv_path: str | None = None) -> str:
    """Finds the IEX_Market_Data.csv file from user argument or standard locations."""
    if csv_path and os.path.exists(csv_path):
        return csv_path
    for candidate in DEFAULT_CSV_PATHS:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate IEX_Market_Data.csv. Please place it in ./data/ or ./IEXInsiderMCP/ "
        "or pass --data-path /path/to/IEX_Market_Data.csv."
    )


def load_iex_dataset(csv_path: str | None = None) -> pd.DataFrame:
    """
    Loads IEX_Market_Data.csv (289,214 records across DAM, RTM, GDAM) and pivots
    into synchronized 15-minute blocks with DAM, RTM, and GDAM prices, demand, and supply.
    """
    resolved_path = resolve_data_path(csv_path)
    df = pd.read_csv(resolved_path, low_memory=False)

    required_cols = [
        "TYPE",
        "Date",
        "Time_Block",
        "IEX_Demand (GW)",
        "IEX_Supply (GW)",
        "MCP (Rs./kWh)",
        "MCV (GW)",
    ]
    df = df[required_cols].copy()

    for num_col in ["IEX_Demand (GW)", "IEX_Supply (GW)", "MCP (Rs./kWh)", "MCV (GW)"]:
        df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Time_Block", "MCP (Rs./kWh)"]).copy()

    time_parts = df["Time_Block"].astype(str).str.split(":", expand=True)
    df["Hour_Float"] = (
        pd.to_numeric(time_parts[0], errors="coerce")
        + pd.to_numeric(time_parts[1], errors="coerce") / 60.0
    )
    df = df.dropna(subset=["Hour_Float"]).copy()
    df["Block_Index"] = (df["Hour_Float"] * 4).round().astype(int) + 1  # 1 to 96

    piv = df.pivot_table(
        index=["Date", "Block_Index", "Hour_Float"],
        columns="TYPE",
        values=["MCP (Rs./kWh)", "IEX_Demand (GW)", "IEX_Supply (GW)", "MCV (GW)"],
        aggfunc="mean",
    )
    piv.columns = [f"{col[1]}_{col[0].split(' ')[0]}" for col in piv.columns]
    piv = piv.reset_index().sort_values(["Date", "Block_Index"]).reset_index(drop=True)
    piv = piv.dropna(subset=["DAM_MCP", "RTM_MCP"]).reset_index(drop=True)
    return piv

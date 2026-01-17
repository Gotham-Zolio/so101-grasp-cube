#!/usr/bin/env python3
"""Debug script to inspect data format from Parquet files"""

import pathlib
import pandas as pd
import numpy as np
import os

# Make sure we're in the right directory
print(f"Current directory: {os.getcwd()}")

task_dir = pathlib.Path("real_data/lift")
parquet_dir = task_dir / "data" / "chunk-000"

print(f"Looking for parquet files in: {parquet_dir.absolute()}")
print(f"Directory exists: {parquet_dir.exists()}")

# Find parquet files
parquet_files = sorted(parquet_dir.glob("file-*.parquet"))
print(f"Found {len(parquet_files)} parquet files")

if parquet_files:
    # Load first parquet file
    df = pd.read_parquet(parquet_files[0])
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check first row
    print(f"\nFirst row:")
    first_row = df.iloc[0]
    for col in df.columns:
        val = first_row[col]
        if isinstance(val, np.ndarray):
            print(f"  {col}: ndarray with shape {val.shape}, dtype {val.dtype}")
            if val.size <= 10:
                print(f"    values: {val}")
        else:
            print(f"  {col}: {type(val).__name__} = {val}")
    
    # Check image column
    if "observation.images.front" in df.columns:
        img = first_row["observation.images.front"]
        if isinstance(img, np.ndarray):
            print(f"\nImage array details:")
            print(f"  Shape: {img.shape}")
            print(f"  Dtype: {img.dtype}")
            print(f"  Min/Max: {img.min()}/{img.max()}")
    
    # Check state
    if "observation.state" in df.columns:
        state = first_row["observation.state"]
        if isinstance(state, np.ndarray):
            print(f"\nState array details:")
            print(f"  Shape: {state.shape}")
            print(f"  Dtype: {state.dtype}")
            print(f"  Values: {state}")
    
    # Check action
    if "action" in df.columns:
        action = first_row["action"]
        if isinstance(action, np.ndarray):
            print(f"\nAction array details:")
            print(f"  Shape: {action.shape}")
            print(f"  Dtype: {action.dtype}")
            print(f"  Values: {action}")

"""
Debug image loading from parquet
"""
import sys
sys.path.insert(0, '/home/gotham/shared/so101-grasp-cube')

import pathlib
import glob
import pandas as pd
import pyarrow.parquet as pq

# Check parquet structure
task_dir = pathlib.Path("/home/gotham/shared/so101-grasp-cube/real_data/lift")
parquet_files = sorted(glob.glob(str(task_dir / "data" / "chunk-*" / "*.parquet")))

if parquet_files:
    pfile = parquet_files[0]
    print(f"Reading parquet file: {pfile}")
    
    # Read with pyarrow to see columns
    table = pq.read_table(pfile)
    print(f"\nParquet columns: {table.column_names}")
    
    # Convert to pandas to inspect data
    df = table.to_pandas()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Column dtypes:\n{df.dtypes}")
    
    # Check a sample row
    print(f"\n" + "="*60)
    print("FIRST ROW CONTENT")
    print("="*60)
    row = df.iloc[0]
    for col in df.columns:
        val = row[col]
        if isinstance(val, (list, tuple)):
            print(f"{col:30s}: length={len(val)}")
        elif hasattr(val, '__len__') and not isinstance(val, str):
            try:
                print(f"{col:30s}: shape={val.shape if hasattr(val, 'shape') else len(val)}")
            except:
                print(f"{col:30s}: {type(val)}")
        else:
            print(f"{col:30s}: {type(val).__name__} = {str(val)[:50]}")

"""
检查 Parquet 文件的列结构
"""
import pathlib
import pyarrow.parquet as pq
import pandas as pd

task = "lift"
task_dir = pathlib.Path(f"real_data/{task}")
parquet_files = sorted(list((task_dir / "data").glob("chunk-*/file-*.parquet")))

if not parquet_files:
    parquet_files = sorted(list((task_dir / "data").glob("chunk-*/*.parquet")))

print(f"Found {len(parquet_files)} parquet files")

if parquet_files:
    pq_file = parquet_files[0]
    print(f"\nInspecting: {pq_file}")
    
    table = pq.read_table(str(pq_file))
    df = table.to_pandas()
    
    print(f"\nShape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = str(df[col].dtype)
        sample_value = df[col].iloc[0]
        print(f"  {i}: {col:50s} | dtype: {dtype:15s} | sample: {sample_value}")

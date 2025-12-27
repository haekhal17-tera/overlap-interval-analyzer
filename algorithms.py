import argparse
import pandas as pd
import numpy as np

from algorithms import (
    sort_intervals,
    count_overlaps_iterative_sorted,
    count_overlaps_recursive_divconq_sorted,
    median_runtime_ms
)

def load_intervals(csv_path: str, start_col: str, end_col: str, max_hours: float = 24.0):
    df = pd.read_csv(csv_path, usecols=[start_col, end_col], low_memory=False)
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    df = df.dropna(subset=[start_col, end_col])
    df = df[df[start_col] < df[end_col]]

    dur_s = (df[end_col] - df[start_col]).dt.total_seconds()
    df = df[dur_s <= max_hours * 3600]

    start_ns = df[start_col].values.astype("datetime64[ns]").astype("int64")
    end_ns = df[end_col].values.astype("datetime64[ns]").astype("int64")
    intervals = np.column_stack([start_ns, end_ns])
    return sort_intervals(intervals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path ke CSV data")
    ap.add_argument("--start_col", default="lpep_pickup_datetime")
    ap.add_argument("--end_col", default="lpep_dropoff_datetime")
    ap.add_argument("--sizes", default="10000,20000,40000,80000,160000")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max_hours", type=float, default=24.0)
    args = ap.parse_args()

    sorted_all = load_intervals(args.csv, args.start_col, args.end_col, max_hours=args.max_hours)
    print("Total intervals after clean:", len(sorted_all))

    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip().isdigit()]
    sizes = sorted(set([n for n in sizes if 0 < n <= len(sorted_all)]))

    rng = np.random.default_rng(42)

    print("\nBenchmark Results (deteksi overlap tanpa sorting)")
    print("n, iter_ms, rec_ms, iter_overlap, rec_overlap")

    for n in sizes:
        idx = rng.choice(len(sorted_all), size=n, replace=False)
        subset = sort_intervals(sorted_all[idx])  # subset harus terurut

        iter_overlap = count_overlaps_iterative_sorted(subset)
        rec_overlap = count_overlaps_recursive_divconq_sorted(subset)

        iter_ms = median_runtime_ms(count_overlaps_iterative_sorted, subset, repeat=args.repeat)
        rec_ms = median_runtime_ms(count_overlaps_recursive_divconq_sorted, subset, repeat=args.repeat)

        print(f"{n}, {iter_ms:.4f}, {rec_ms:.4f}, {iter_overlap}, {rec_overlap}")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from algorithms import (
    count_overlaps_iterative,
    count_overlaps_recursive_divconq,
    median_runtime_ms
)

st.set_page_config(page_title="Overlap Interval Analyzer", layout="wide")

# =========================
# Helpers
# =========================
def parse_manual_intervals(text: str):
    """
    Format per baris:
      start,end

    Contoh:
      2024-01-01 10:00,2024-01-01 10:20
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Format salah: '{line}'. Harus 'start,end'")
        rows.append(parts)

    df = pd.DataFrame(rows, columns=["start", "end"])
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df = df.dropna()
    df = df[df["start"] < df["end"]]

    start_ns = df["start"].values.astype("datetime64[ns]").astype("int64")
    end_ns = df["end"].values.astype("datetime64[ns]").astype("int64")
    intervals = np.column_stack([start_ns, end_ns])
    return intervals, df

def load_csv_intervals(file, pickup_col: str, dropoff_col: str, max_hours: float = 24.0):
    df = pd.read_csv(file, usecols=[pickup_col, dropoff_col], low_memory=False)
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
    df = df.dropna(subset=[pickup_col, dropoff_col])
    df = df[df[pickup_col] < df[dropoff_col]]

    # filter durasi ekstrem
    dur_s = (df[dropoff_col] - df[pickup_col]).dt.total_seconds()
    df = df[dur_s <= max_hours * 3600]

    start_ns = df[pickup_col].values.astype("datetime64[ns]").astype("int64")
    end_ns = df[dropoff_col].values.astype("datetime64[ns]").astype("int64")
    intervals = np.column_stack([start_ns, end_ns])
    return intervals, df

# =========================
# UI
# =========================
st.title("Aplikasi Sederhana — Deteksi Overlapping Interval (Iteratif vs Rekursif)")
st.write(
    "Studi kasus: mendeteksi interval perjalanan taksi yang saling tumpang-tindih berdasarkan waktu pickup dan dropoff. "
    "Aplikasi ini mendukung input manual dan CSV (misalnya NYC Green Taxi)."
)

algo_choice = st.radio(
    "Pilih Algoritma",
    ["Iteratif (Sweep)", "Rekursif (Divide & Conquer)"],
    horizontal=True
)

repeat = st.number_input(
    "Jumlah iterasi pengujian runtime (repeat, median diambil)",
    min_value=1, max_value=50, value=5, step=1
)

algo_fn = count_overlaps_iterative if "Iteratif" in algo_choice else count_overlaps_recursive_divconq

tab_manual, tab_csv, tab_bench = st.tabs(["Input Manual", "Input CSV", "Benchmark & Grafik"])

# -------------------------
# Manual input
# -------------------------
with tab_manual:
    st.subheader("Mode 1 — Input Manual")
    st.caption("Format: `start,end` per baris (datetime).")

    sample_text = (
        "2024-01-01 10:00,2024-01-01 10:20\n"
        "2024-01-01 10:10,2024-01-01 10:40\n"
        "2024-01-01 10:50,2024-01-01 11:10\n"
        "2024-01-01 11:00,2024-01-01 11:20\n"
    )
    text = st.text_area("Masukkan interval:", value=sample_text, height=170)

    if st.button("Jalankan (Manual)"):
        try:
            intervals, df_used = parse_manual_intervals(text)
            if len(intervals) == 0:
                st.warning("Tidak ada interval valid. Pastikan format benar dan start < end.")
            else:
                overlap_count = algo_fn(intervals)
                runtime_ms = median_runtime_ms(algo_fn, intervals, repeat=repeat)

                st.success(f"Jumlah overlap terdeteksi: **{overlap_count}**")
                st.info(f"Runtime median ({repeat} kali): **{runtime_ms:.4f} ms**")
                st.write("Interval valid yang digunakan:")
                st.dataframe(df_used)
        except Exception as e:
            st.error(str(e))

# -------------------------
# CSV input
# -------------------------
with tab_csv:
    st.subheader("Mode 2 — Upload CSV")
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])

    col1, col2, col3 = st.columns(3)
    with col1:
        pickup_col = st.text_input("Kolom start (pickup)", value="lpep_pickup_datetime")
    with col2:
        dropoff_col = st.text_input("Kolom end (dropoff)", value="lpep_dropoff_datetime")
    with col3:
        max_hours = st.number_input("Filter durasi maksimum (jam)", min_value=1.0, max_value=72.0, value=24.0, step=1.0)

    limit_n = st.number_input("Batasi jumlah baris (0=tanpa batas)", min_value=0, max_value=2_000_000, value=0, step=10_000)

    if uploaded is not None:
        if st.button("Jalankan (CSV)"):
            intervals, df_used = load_csv_intervals(uploaded, pickup_col, dropoff_col, max_hours=max_hours)

            if limit_n and limit_n < len(intervals):
                intervals = intervals[:limit_n]
                df_used = df_used.iloc[:limit_n].copy()

            overlap_count = algo_fn(intervals)
            runtime_ms = median_runtime_ms(algo_fn, intervals, repeat=repeat)

            st.success(f"Jumlah overlap terdeteksi: **{overlap_count}**")
            st.info(f"Runtime median ({repeat} kali): **{runtime_ms:.3f} ms**")
            st.write(f"Jumlah interval setelah clean: **{len(intervals)}**")
            st.write("Preview data (20 baris pertama):")
            st.dataframe(df_used.head(20))

# -------------------------
# Benchmark
# -------------------------
with tab_bench:
    st.subheader("Mode 3 — Benchmark & Grafik (CSV wajib)")
    st.caption("Gunakan benchmark untuk membuat grafik running time vs ukuran input.")

    uploaded_b = st.file_uploader("Upload CSV untuk Benchmark", type=["csv"], key="bench_csv")
    b_pickup = st.text_input("Kolom start", value="lpep_pickup_datetime", key="bp")
    b_dropoff = st.text_input("Kolom end", value="lpep_dropoff_datetime", key="bd")
    sizes_text = st.text_input("Ukuran benchmark (pisahkan koma)", value="10000,20000,40000,80000,160000")
    repeat_b = st.number_input("Repeat benchmark", min_value=1, max_value=20, value=3, step=1, key="rb")

    if uploaded_b is not None and st.button("Jalankan Benchmark"):
        intervals, _ = load_csv_intervals(uploaded_b, b_pickup, b_dropoff, max_hours=24.0)

        sizes = [int(x.strip()) for x in sizes_text.split(",") if x.strip().isdigit()]
        sizes = [n for n in sizes if 0 < n <= len(intervals)]
        sizes = sorted(set(sizes))

        if len(sizes) == 0:
            st.warning("Ukuran benchmark kosong atau lebih besar dari jumlah data setelah clean.")
        else:
            rng = np.random.default_rng(42)
            rows = []
            for n in sizes:
                idx = rng.choice(len(intervals), size=n, replace=False)
                subset = intervals[idx]

                t_iter = median_runtime_ms(count_overlaps_iterative, subset, repeat=repeat_b)
                t_rec = median_runtime_ms(count_overlaps_recursive_divconq, subset, repeat=repeat_b)

                rows.append((n, t_iter, t_rec))

            df_res = pd.DataFrame(rows, columns=["n", "iter_ms", "rec_ms"])
            st.dataframe(df_res)

            fig = plt.figure()
            plt.plot(df_res["n"], df_res["iter_ms"], marker="o")
            plt.plot(df_res["n"], df_res["rec_ms"], marker="o")
            plt.xlabel("Ukuran input (n interval)")
            plt.ylabel("Runtime (ms)")
            plt.title("Benchmark Running Time — Iteratif vs Rekursif")
            plt.legend(["Iteratif (Sweep)", "Rekursif (Divide & Conquer)"])
            plt.xscale("log")
            st.pyplot(fig)

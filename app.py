import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from algorithms import (
    sort_intervals,
    count_overlaps_iterative_sorted,
    count_overlaps_recursive_divconq_sorted,
    median_runtime_ms,
)

# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Overlap Interval Analyzer", layout="wide")
DATASET_FILENAME = "taxi_tripdata.csv"  # sesuai file yang kamu upload

# ======================================================
# DATA LOADER
# ======================================================
def load_intervals_from_csv(file_or_path, start_col: str, end_col: str, max_hours: float):
    df = pd.read_csv(file_or_path, usecols=[start_col, end_col], low_memory=False)
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")

    df = df.dropna(subset=[start_col, end_col])
    df = df[df[start_col] < df[end_col]]

    # filter durasi ekstrem
    dur_sec = (df[end_col] - df[start_col]).dt.total_seconds()
    df = df[dur_sec <= max_hours * 3600]

    start_ns = df[start_col].values.astype("datetime64[ns]").astype("int64")
    end_ns = df[end_col].values.astype("datetime64[ns]").astype("int64")

    intervals = np.column_stack([start_ns, end_ns])
    df_disp = df.rename(columns={start_col: "start", end_col: "end"})

    return intervals, df_disp[["start", "end"]]

# ======================================================
# HEADER
# ======================================================
st.title("Overlap Interval Analyzer — Iteratif vs Rekursif")

st.markdown(
    """
Aplikasi ini menganalisis **kompleksitas dan running time deteksi interval overlap**
dengan dua pendekatan sekaligus:

- **Iteratif (Sweep Line)** → O(n)
- **Rekursif (Divide & Conquer)** → O(n log n)

Dataset: **`taxi_tripdata.csv`** (trip records).  
Input manual yang kamu atur adalah **ukuran input (n)** untuk benchmark.
"""
)

with st.expander("Definisi overlap (untuk laporan)", expanded=False):
    st.markdown(
        """
Interval dianggap **overlap** jika setelah diurutkan berdasarkan `start`,
saat memindai interval ke-i berlaku:

> `start_i < current_max_end`

Output yang dihitung adalah **jumlah interval yang terdeteksi overlap**,  
bukan jumlah pasangan overlap.
"""
    )

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.header("Pengaturan")

max_hours = st.sidebar.number_input(
    "Filter durasi maksimum (jam)",
    min_value=1.0, max_value=72.0, value=24.0
)

limit_preview = st.sidebar.number_input(
    "Preview baris (tabel data)",
    min_value=5, max_value=200, value=20, step=5
)

# ======================================================
# LOAD DATA
# ======================================================
st.subheader("Dataset")

data_source = st.radio(
    "Sumber data:",
    ["File lokal (taxi_tripdata.csv)", "Upload CSV"],
    horizontal=True
)

col1, col2 = st.columns(2)
with col1:
    start_col = st.text_input("Kolom start", value="lpep_pickup_datetime")
with col2:
    end_col = st.text_input("Kolom end", value="lpep_dropoff_datetime")

csv_obj = None
if data_source.startswith("File lokal"):
    if os.path.exists(DATASET_FILENAME):
        st.success(f"Dataset ditemukan: `{DATASET_FILENAME}`")
        csv_obj = DATASET_FILENAME
    else:
        st.error(
            f"File `{DATASET_FILENAME}` tidak ditemukan.\n\n"
            "Letakkan file tersebut di root project (sejajar app.py)."
        )
else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        csv_obj = uploaded

if csv_obj is None:
    st.stop()

intervals, df_display = load_intervals_from_csv(csv_obj, start_col, end_col, max_hours)
sorted_intervals = sort_intervals(intervals)

st.info(f"Jumlah interval setelah preprocessing: **{len(sorted_intervals)}**")
st.dataframe(df_display.head(int(limit_preview)))

# ======================================================
# SINGLE RUN (OPSIONAL) - hasil dua metode langsung
# ======================================================
st.subheader("Deteksi Overlap (langsung dua metode)")

run_repeat = st.number_input(
    "Repeat runtime (ambil median) untuk deteksi full-data",
    min_value=1, max_value=50, value=5
)

if st.button("Jalankan Deteksi (Full Data)"):
    subset = sorted_intervals  # sudah terurut

    iter_overlap = count_overlaps_iterative_sorted(subset)
    rec_overlap = count_overlaps_recursive_divconq_sorted(subset)

    iter_ms = median_runtime_ms(count_overlaps_iterative_sorted, subset, repeat=run_repeat)
    rec_ms = median_runtime_ms(count_overlaps_recursive_divconq_sorted, subset, repeat=run_repeat)

    st.table(pd.DataFrame([{
        "n": len(subset),
        "iter_overlap": iter_overlap,
        "iter_ms": round(iter_ms, 4),
        "rec_overlap": rec_overlap,
        "rec_ms": round(rec_ms, 4),
    }]))

# ======================================================
# BENCHMARK (n manual) - tabel langsung dua metode
# ======================================================
st.subheader("Benchmark — Ukuran Input Manual (n)")

st.markdown(
    """
Masukkan **ukuran input (n)** secara manual, misalnya:

`1,10,20,50,100,200,500,1000,5000,10000`
"""
)

sizes_text = st.text_input(
    "Daftar ukuran input (pisahkan koma)",
    value="1,10,20,50,100,200,500,1000,5000,10000"
)

sample_mode = st.radio(
    "Metode pengambilan subset",
    ["Prefix (ambil n pertama)", "Random (acak, lebih fair)"],
    horizontal=True
)

repeat_bench = st.number_input(
    "Repeat benchmark",
    min_value=1, max_value=20, value=3
)

if st.button("Jalankan Benchmark"):
    try:
        sizes = sorted({
            int(x.strip()) for x in sizes_text.split(",")
            if x.strip().isdigit() and int(x.strip()) > 0
        })

        sizes = [n for n in sizes if n <= len(sorted_intervals)]
        if not sizes:
            st.error("Ukuran input tidak valid atau melebihi jumlah data.")
            st.stop()

        rng = np.random.default_rng(42)
        rows = []

        for n in sizes:
            if sample_mode.startswith("Prefix"):
                subset = sorted_intervals[:n]
            else:
                idx = rng.choice(len(sorted_intervals), size=n, replace=False)
                subset = sort_intervals(sorted_intervals[idx])

            # hitung overlap (cek konsistensi hasil)
            iter_overlap = count_overlaps_iterative_sorted(subset)
            rec_overlap = count_overlaps_recursive_divconq_sorted(subset)

            # runtime median (ms)
            iter_ms = median_runtime_ms(count_overlaps_iterative_sorted, subset, repeat=repeat_bench)
            rec_ms = median_runtime_ms(count_overlaps_recursive_divconq_sorted, subset, repeat=repeat_bench)

            rows.append((n, iter_overlap, iter_ms, rec_overlap, rec_ms))

        df_result = pd.DataFrame(
            rows,
            columns=["n", "iter_overlap", "iter_ms", "rec_overlap", "rec_ms"]
        )

        st.dataframe(df_result)

        # Plot runtime
        fig = plt.figure()
        plt.plot(df_result["n"], df_result["iter_ms"], marker="o")
        plt.plot(df_result["n"], df_result["rec_ms"], marker="o")
        plt.xlabel("Ukuran input (n)")
        plt.ylabel("Runtime deteksi overlap (ms)")
        plt.title("Perbandingan Running Time — Iteratif vs Rekursif")
        plt.legend(["Iteratif — O(n)", "Rekursif — O(n log n)"])
        plt.xscale("log")
        st.pyplot(fig)

        st.success("Benchmark selesai.")

    except Exception as e:
        st.error(str(e))

from __future__ import annotations
import numpy as np
import time

def sort_intervals(intervals: np.ndarray) -> np.ndarray:
    """
    intervals: np.ndarray shape (n,2) => [start_ns, end_ns]
    return: sorted by start then end (ascending)
    """
    if len(intervals) == 0:
        return intervals
    return intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))]

def count_overlaps_iterative(intervals: np.ndarray) -> int:
    """
    Overlap definition (stable for benchmarking):
      interval i dianggap overlap jika start_i < current_max_end
      saat menyapu interval yang sudah diurutkan berdasarkan start time.

    Output:
      jumlah interval yang terdeteksi overlap (bukan jumlah pasangan overlap).
    Complexity:
      - Sorting: O(n log n)
      - Sweep:   O(n)
      Total: O(n log n), tapi inti overlap detection sweep = O(n).
    """
    if len(intervals) == 0:
        return 0

    iv = sort_intervals(intervals)
    overlaps = 0
    current_end = iv[0, 1]

    for i in range(1, len(iv)):
        s, e = iv[i]
        if s < current_end:
            overlaps += 1
            if e > current_end:
                current_end = e
        else:
            current_end = e

    return overlaps

def count_overlaps_recursive_divconq(intervals: np.ndarray) -> int:
    """
    Divide & Conquer overlap detection.
    Kita tetap sort dulu agar interval kanan punya start >= interval kiri (secara global).

    Intuisi:
      - hitung overlap di kiri
      - hitung overlap di kanan
      - hitung overlap silang dengan scanning bagian kanan menggunakan max_end dari kiri

    Complexity (setelah sorting):
      T(n) = 2T(n/2) + O(n) => O(n log n)

    Output:
      jumlah interval yang terdeteksi overlap (definisi sama seperti iteratif).
    """
    if len(intervals) == 0:
        return 0

    iv = sort_intervals(intervals)

    def solve(lo: int, hi: int):
        n = hi - lo
        if n <= 1:
            if n == 1:
                return 0, iv[lo, 1]  # (count, max_end)
            return 0, -np.inf

        mid = lo + n // 2
        left_count, left_maxend = solve(lo, mid)
        right_count, right_maxend = solve(mid, hi)

        cross = 0
        max_end_so_far = left_maxend
        for i in range(mid, hi):
            s, e = iv[i]
            if s < max_end_so_far:
                cross += 1
            if e > max_end_so_far:
                max_end_so_far = e

        return left_count + right_count + cross, max(max_end_so_far, right_maxend)

    total, _ = solve(0, len(iv))
    return total

def median_runtime_ms(fn, intervals: np.ndarray, repeat: int = 5) -> float:
    """
    Mengukur runtime median (ms) untuk stabilitas.
    """
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn(intervals)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.median(times))

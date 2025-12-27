from __future__ import annotations
import numpy as np
import time

# =========================================================
# Helpers
# =========================================================
def sort_intervals(intervals: np.ndarray) -> np.ndarray:
    """
    intervals: np.ndarray shape (n,2) => [start_ns, end_ns]
    return: sorted by start then end (ascending)
    """
    if intervals is None or intervals.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return intervals[np.lexsort((intervals[:, 1], intervals[:, 0]))]

def median_runtime_ms(fn, *args, repeat: int = 5) -> float:
    """
    Mengukur runtime median (ms) untuk stabilitas.
    """
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.median(times))

# =========================================================
# Overlap detection (ASSUME SORTED)
# =========================================================
def count_overlaps_iterative_sorted(sorted_intervals: np.ndarray) -> int:
    """
    Iteratif sweep O(n) pada data yang SUDAH terurut.
    Overlap jika start_i < current_max_end.
    Output: jumlah interval yang overlap (bukan pasangan).
    """
    if sorted_intervals is None or len(sorted_intervals) == 0:
        return 0

    overlaps = 0
    current_end = int(sorted_intervals[0, 1])

    for i in range(1, len(sorted_intervals)):
        s = int(sorted_intervals[i, 0])
        e = int(sorted_intervals[i, 1])

        if s < current_end:
            overlaps += 1
            if e > current_end:
                current_end = e
        else:
            current_end = e

    return overlaps

def count_overlaps_recursive_divconq_sorted(sorted_intervals: np.ndarray) -> int:
    """
    Rekursif divide & conquer pada data yang SUDAH terurut.
    T(n)=2T(n/2)+O(n) => O(n log n)
    """
    if sorted_intervals is None or len(sorted_intervals) == 0:
        return 0

    iv = sorted_intervals

    def solve(lo: int, hi: int):
        n = hi - lo
        if n <= 1:
            if n == 1:
                return 0, int(iv[lo, 1])
            return 0, -10**30

        mid = lo + n // 2
        left_count, left_maxend = solve(lo, mid)
        right_count, right_maxend = solve(mid, hi)

        cross = 0
        max_end_so_far = left_maxend
        for i in range(mid, hi):
            s = int(iv[i, 0])
            e = int(iv[i, 1])
            if s < max_end_so_far:
                cross += 1
            if e > max_end_so_far:
                max_end_so_far = e

        return left_count + right_count + cross, max(max_end_so_far, right_maxend)

    total, _ = solve(0, len(iv))
    return total

# =========================================================
# Wrappers (include sorting) - optional
# =========================================================
def count_overlaps_iterative(intervals: np.ndarray) -> int:
    return count_overlaps_iterative_sorted(sort_intervals(intervals))

def count_overlaps_recursive_divconq(intervals: np.ndarray) -> int:
    return count_overlaps_recursive_divconq_sorted(sort_intervals(intervals))

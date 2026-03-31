import numpy as np
from numba import njit
import time, statistics
import matplotlib.pyplot as plt
from pathlib import Path
from dask import delayed
import dask
from dask.distributed import Client
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):
    zr = 0.0
    zi = 0.0
    for i in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > 4.0:
            return i
        zi = 2.0 * zr * zi + c_imag
        zr = zr2 - zi2 + c_real
    return max_iter


@njit(cache=True)
def mandelbrot_chunk(row_start, row_end, N, xmin, xmax, ymin, ymax, max_iter):
    out = np.empty((row_end - row_start, N), dtype=np.int32)
    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / N

    for r in range(row_end - row_start):
        c_imag = ymin + (r + row_start) * dy
        for col in range(N):
            c_real = xmin + col * dx
            out[r, col] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return out


def mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter=100):
    return mandelbrot_chunk(0, N, N, xmin, xmax, ymin, ymax, max_iter)


def mandelbrot_dask(N, xmin, xmax, ymin, ymax, max_iter=100, n_chunks=16):
    chunk_size = max(1, N // n_chunks)
    chunks = []

    row = 0
    while row < N:
        row_end = min(row + chunk_size, N)
        chunks.append((row, row_end, N, xmin, xmax, ymin, ymax, max_iter))
        row = row_end

    tasks = [delayed(mandelbrot_chunk)(*chunk) for chunk in chunks]
    parts = dask.compute(*tasks)
    return np.vstack(parts)


def warmup(xmin, xmax, ymin, ymax):
    mandelbrot_chunk(0, 8, 8, xmin, xmax, ymin, ymax, 10)


if __name__ == "__main__":
    # =========================
    # PROBLEM SETUP
    # =========================
    N = 4096
    max_iter = 100
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    # =========================
    # DASK SETUP
    # =========================
    client = Client("tcp://10.92.1.130:8786")

    versions = client.run(lambda: __import__('dask').__version__)
    print(versions)

    # warm-up ALL workers (Numba JIT)
    client.run(warmup, xmin, xmax, ymin, ymax)

    # warm-up
    mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter)
    mandelbrot_dask(N, xmin, xmax, ymin, ymax, max_iter, n_chunks=8)

    # =========================
    # SERIAL BASELINE
    # =========================
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter)
        times.append(time.perf_counter() - t0)

    t_serial = statistics.median(times)
    print(f"\nSerial baseline: {t_serial:.3f}s\n")

    # =========================
    # VERIFY CORRECTNESS
    # =========================
    result_serial = mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter)
    result_dask = mandelbrot_dask(N, xmin, xmax, ymin, ymax, max_iter, n_chunks=8)

    print("Match:", np.array_equal(result_serial, result_dask))

    # =========================
    # M2: CHUNK SWEEP
    # =========================
    chunk_values = [2, 4, 8, 16, 32, 64, 128, 256]
    results = []

    print("\nn_chunks | time (s) | vs1x | Speedup | LIF")
    print("------------------------------------------------")

    baseline = None

    for n_chunks in chunk_values:
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            mandelbrot_dask(N, xmin, xmax, ymin, ymax, max_iter, n_chunks=n_chunks)
            times.append(time.perf_counter() - t0)

        t_dask = statistics.median(times)
        if baseline is None:
            baseline = t_dask
        vs1x = t_dask / baseline
        speedup = t_serial / t_dask
        lif = 4 * (t_dask / t_serial) - 1

        results.append((n_chunks, t_dask, vs1x, speedup, lif))

        print(f"{n_chunks:8d} | {t_dask:8.3f} | {vs1x:6.2f} | {speedup:7.2f}x | {lif:6.2f}")

    # =========================
    # FIND BEST
    # =========================
    best = min(results, key=lambda x: x[1])
    best_lif = min(results, key=lambda x: x[4])

    print("\nBest result:")
    print(f"n_chunks_optimal = {best[0]}")
    print(f"t_min = {best[1]:.3f} s")
    print(f"speedup = {best[3]:.2f}x")
    print(f"LIF_min = {best_lif[4]:.2f}")

    # =========================
    # CLEAN UP
    # =========================
    plt.plot(chunk_values, [r[1] for r in results])
    plt.xscale("log")
    plt.xlabel("n_chunks")
    plt.ylabel("Time (s)")
    plt.savefig("dask_chunk_sweep.png")
    plt.show()

    client.close()

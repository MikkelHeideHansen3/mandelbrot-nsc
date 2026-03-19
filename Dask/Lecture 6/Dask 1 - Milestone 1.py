import numpy as np
from numba import njit
import time, statistics
import matplotlib.pyplot as plt
from pathlib import Path
from dask import delayed
import dask
from dask.distributed import Client, LocalCluster
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

@njit(cache=True)
def mandelbrot_pixel(c_real, c_imag, max_iter):

    zr = 0.0
    zi = 0.0

    for i in range(max_iter):

        zr2 = zr*zr
        zi2 = zi*zi

        if zr2 + zi2 > 4.0:
            return i

        zi = 2.0*zr*zi + c_imag
        zr = zr2 - zi2 + c_real

    return max_iter

@njit(cache=True)
def mandelbrot_chunk(row_start, row_end,
                     N,
                     xmin, xmax, ymin, ymax,
                     max_iter):

    out = np.empty((row_end - row_start, N), dtype=np.int32)

    dx = (xmax - xmin) / N
    dy = (ymax - ymin) / N

    for r in range(row_end - row_start):

        c_imag = ymin + (r + row_start) * dy

        for col in range(N):

            c_real = xmin + col * dx

            out[r, col] = mandelbrot_pixel(
                c_real,
                c_imag,
                max_iter
            )

    return out

def mandelbrot_serial(N,
                      xmin, xmax,
                      ymin, ymax,
                      max_iter=100):

    return mandelbrot_chunk(
        0, N,
        N,
        xmin, xmax,
        ymin, ymax,
        max_iter
    )

def mandelbrot_dask(
        N,
        xmin, xmax,
        ymin, ymax,
        max_iter=100,
        n_chunks=16):

    chunk_size = max(1, N // n_chunks)

    chunks = []
    row = 0

    while row < N:
        row_end = min(row + chunk_size, N)

        chunks.append(
            (row, row_end, N,
             xmin, xmax,
             ymin, ymax,
             max_iter)
        )

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

    N = 1024
    max_iter = 100

    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5


    # =========================
    # SERIAL BASELINE
    # =========================

    cluster = LocalCluster(n_workers=4, threads_per_worker=1)
    client = Client(cluster)

    client.run(warmup, xmin, xmax, ymin, ymax)


    # warm-up
    mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter)

    times = []

    for _ in range(5):

        t0 = time.perf_counter()

        mandelbrot_serial(
            N,
            xmin, xmax,
            ymin, ymax,
            max_iter
        )

        times.append(time.perf_counter() - t0)

    t_serial = statistics.median(times)

    print(f"\nSerial baseline: {t_serial:.3f}s\n")

    # Verify result matches serial output
    result_serial = mandelbrot_serial(
    N, xmin, xmax, ymin, ymax, max_iter
)

    result_dask = mandelbrot_dask(
    N, xmin, xmax, ymin, ymax,
    max_iter,
    n_chunks=8
)

    print("Match:", np.array_equal(result_serial, result_dask))


    times = []

    for _ in range(5):
        t0 = time.perf_counter()

        mandelbrot_dask(
            N, xmin, xmax, ymin, ymax,
            max_iter,
            n_chunks=8
        )

        times.append(time.perf_counter() - t0)

    t_dask = statistics.median(times)

    print(f"\nDask results:")
    print(f"Time: {t_dask:.3f}s")
    print(f"Speedup: {t_serial / t_dask:.2f}x")

    client.close()
    cluster.close() 
    
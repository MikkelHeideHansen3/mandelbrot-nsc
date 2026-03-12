import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics
import matplotlib.pyplot as plt
from pathlib import Path

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

def _worker(args):
    return mandelbrot_chunk(*args)

def mandelbrot_parallel(
        N,
        xmin, xmax,
        ymin, ymax,
        max_iter=100,
        n_workers=4,
        n_chunks=None):

    if n_chunks is None:
        n_chunks = n_workers

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

    with Pool(processes=n_workers) as pool:

        tiny = [(0, 8, 8, xmin, xmax, ymin, ymax, max_iter)]
        pool.map(_worker, tiny)   # warm-up

        parts = pool.map(_worker, chunks)

    return np.vstack(parts)



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

    # warm-up
    mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter)

    times = []

    for _ in range(3):

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

    result_parallel = mandelbrot_parallel(
    N, xmin, xmax, ymin, ymax,
    max_iter,
    n_workers=os.cpu_count(),
    n_chunks=os.cpu_count()*4
)

    print("Match:", np.array_equal(result_serial, result_parallel))

    # =========================
    # WORKER SWEEP
    # =========================

    for n_workers in range(1, os.cpu_count() + 1):

        chunk_size = max(1, N // n_workers)

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


        with Pool(processes=n_workers) as pool:

            # warm-up
            tiny = [(0, 8, 8, xmin, xmax, ymin, ymax, max_iter)]
            pool.map(_worker, tiny)

            times = []

            for _ in range(3):

                t0 = time.perf_counter()

                parts = pool.map(_worker, chunks)
                np.vstack(parts)

                times.append(time.perf_counter() - t0)

        t_par = statistics.median(times)

        speedup = t_serial / t_par
        efficiency = speedup / n_workers

        
        print(
            f"\n{n_workers:2d} workers: "
            f"{t_par:.3f}s   "
            f"speedup={speedup:.2f}x   "
            f"eff={efficiency*100:.0f}%"
        )


    # =========================
    # CHUNK SWEEP
    # =========================

    print("\n=== CHUNK SWEEP ===\n")

    n_workers = os.cpu_count()

    with Pool(processes=n_workers) as pool:

        # warm-up
        tiny = [(0, 8, 8, xmin, xmax, ymin, ymax, max_iter)]
        pool.map(_worker, tiny)

        for factor in [1, 2, 4, 8, 16]:

            n_chunks = factor * n_workers

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

            times = []

            for _ in range(3):

                t0 = time.perf_counter()

                parts = pool.map(_worker, chunks)
                np.vstack(parts)

                times.append(time.perf_counter() - t0)

            t_par = statistics.median(times)

            speedup = t_serial / t_par
            LIF = n_workers * t_par / t_serial - 1

            print(
                f"{n_chunks:4d} chunks: "
                f"{t_par:.3f}s   "
                f"speedup={speedup:.2f}x   "
                f"LIF={LIF:.2f}"
            )
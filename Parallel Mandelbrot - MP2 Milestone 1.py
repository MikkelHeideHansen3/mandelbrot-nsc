import numpy as np
from numba import njit
from multiprocessing import Pool
import time, os, statistics
import matplotlib.pyplot as plt
from pathlib import Path

@njit
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

@njit
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

if __name__ == "__main__":

    N = 1024
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    max_iter = 100

    # warmup
    mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter)

    t0 = time.perf_counter()

    result = mandelbrot_serial(
        N, xmin, xmax,
        ymin, ymax,
        max_iter
    )

    t1 = time.perf_counter()

    print(f"Serial Mandelbrot time: {t1 - t0:.3f}s")

    plt.imshow(result, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.colorbar()
    plt.show()
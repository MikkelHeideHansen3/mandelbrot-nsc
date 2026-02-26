"""
Mandelbrot Set Generator 
Author: Mikkel Heide Hansen
Course: Numerical Scientific Computing (NSC) 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import statistics
from numba import njit

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

def mandelbrot_point(c, max_iter):
    z = 0.0 + 0.0j
    for n in range(max_iter):
        if abs(z) >= 2:
            return n
        z = z*z + c
    return max_iter


def compute_mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    result = np.zeros(C.shape, dtype=int)

    for i in range(height):
        for j in range(width):
            result[i, j] = mandelbrot_point(C[i, j], max_iter)

    return result


def compute_mandelbrot_vectorized(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)

    for n in range(max_iter):
        mask = np.abs(Z) <= 2
        if not mask.any():
            break

        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M


def benchmark(func, *args, n_runs=2):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    median_t = statistics.median(times)
    print(f"Median: {median_t:.4f}s")
    return median_t, result

@njit
def compute_mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):

            zr = 0.0
            zi = 0.0
            cr = x[j]
            ci = y[i]

            for n in range(max_iter):

                if zr*zr + zi*zi >= 4.0:
                    result[i, j] = n
                    break

                temp = zr*zr - zi*zi + cr
                zi = 2.0*zr*zi + ci
                zr = temp

            else:
                result[i, j] = max_iter

    return result

if __name__ == "__main__":

    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    print("Naive version:")
    t_naive, result_naive = benchmark(
        compute_mandelbrot_naive,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    print("\nVectorized version:")
    t_vec, result_vec = benchmark(
        compute_mandelbrot_vectorized,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    print(f"\nSpeedup: {t_naive / t_vec:.2f}x faster")

    # Plot naive
    plt.figure(figsize=(6, 6))
    plt.imshow(result_naive, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.title("Mandelbrot (Naive)")
    plt.colorbar()
    plt.show()

    # Plot vectorized
    plt.figure(figsize=(6, 6))
    plt.imshow(result_vec, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.title("Mandelbrot (NumPy)")
    plt.colorbar()
    plt.show()

    print("\nNumba version:")

    # Warmup (kompilerer)
    compute_mandelbrot_numba(
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    t_numba, result_numba = benchmark(
        compute_mandelbrot_numba,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    print(f"\nSpeedup vs naive: {t_naive / t_numba:.2f}x faster")
    print(f"Speedup vs NumPy: {t_vec / t_numba:.2f}x faster")
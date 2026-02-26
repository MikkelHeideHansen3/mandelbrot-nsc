import numpy as np
import time
import statistics
from numba import njit


# ----------------------------
# NAIVE
# ----------------------------
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
    result = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            c = x[j] + 1j*y[i]
            result[i, j] = mandelbrot_point(c, max_iter)

    return result


# ----------------------------
# NUMPY
# ----------------------------
def compute_mandelbrot_numpy(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j*Y

    Z = np.zeros_like(C)
    M = np.zeros(C.shape, dtype=int)

    for _ in range(max_iter):
        mask = np.abs(Z) <= 2
        if not mask.any():
            break
        Z[mask] = Z[mask]**2 + C[mask]
        M[mask] += 1

    return M


# ----------------------------
# NUMBA (fully compiled)
# ----------------------------
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


# ----------------------------
# Benchmark helper
# ----------------------------
def bench(func, *args, runs=3):
    times = []

    for _ in range(runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)

    return statistics.median(times)


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    # Warm-up Numba (important!)
    compute_mandelbrot_numba(xmin, xmax, ymin, ymax, width, height, max_iter)

    print("Running benchmarks...\n")

    t_naive = bench(compute_mandelbrot_naive,
                    xmin, xmax, ymin, ymax, width, height, max_iter)

    t_numpy = bench(compute_mandelbrot_numpy,
                    xmin, xmax, ymin, ymax, width, height, max_iter)

    t_numba = bench(compute_mandelbrot_numba,
                    xmin, xmax, ymin, ymax, width, height, max_iter)

    # Print table
    print("Implementation   Time (s)   Speedup vs Naive")
    print("----------------------------------------------")
    print(f"Naive            {t_naive:8.3f}     1.0x")
    print(f"NumPy            {t_numpy:8.3f}     {t_naive/t_numpy:6.1f}x")
    print(f"Numba            {t_numba:8.3f}     {t_naive/t_numba:6.1f}x")
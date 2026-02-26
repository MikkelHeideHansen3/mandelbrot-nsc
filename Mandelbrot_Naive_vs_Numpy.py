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


def benchmark(func, *args, n_runs=5):
    """
    Benchmarks a function using:
    - 1 warm-up run (not timed)
    - n_runs timed executions
    - Returns mean and standard deviation
    """

    # Warm-up (important for Numba JIT)
    func(*args)

    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - t0)
    
    median = statistics.median(times)
    std = statistics.stdev(times)

    return median, std

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


@njit
def mandelbrot_numba_f32(xmin, xmax, ymin, ymax,
                         width, height, max_iter=100):

    x = np.linspace(xmin, xmax, width).astype(np.float32)
    y = np.linspace(ymin, ymax, height).astype(np.float32)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):

            zx = np.float32(0.0)
            zy = np.float32(0.0)

            cx = x[j]
            cy = y[i]

            for n in range(max_iter):

                if zx*zx + zy*zy >= np.float32(4.0):
                    result[i, j] = n
                    break

                temp = zx*zx - zy*zy + cx
                zy = np.float32(2.0)*zx*zy + cy
                zx = temp

            else:
                result[i, j] = max_iter

    return result

@njit
def mandelbrot_numba_f64(xmin, xmax, ymin, ymax,
                         width, height, max_iter=100):

    x = np.linspace(xmin, xmax, width).astype(np.float64)
    y = np.linspace(ymin, ymax, height).astype(np.float64)

    result = np.zeros((height, width), dtype=np.int32)

    for i in range(height):
        for j in range(width):

            zx = 0.0
            zy = 0.0

            cx = x[j]
            cy = y[i]

            for n in range(max_iter):

                if zx*zx + zy*zy >= 4.0:
                    result[i, j] = n
                    break

                temp = zx*zx - zy*zy + cx
                zy = 2.0*zx*zy + cy
                zx = temp

            else:
                result[i, j] = max_iter

    return result

if __name__ == "__main__":

    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    # =============================
    # NAIVE
    # =============================
    print("Naive version:")

    # Generate result for plotting
    result_naive = compute_mandelbrot_naive(
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    # Benchmark separately
    mu_naive, sigma_naive = benchmark(
        compute_mandelbrot_naive,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    print(f"Naive: {mu_naive:.4f} ± {sigma_naive:.4f} s")


    # =============================
    # NUMPY VECTORIZED
    # =============================
    print("\nVectorized version:")

    result_vec = compute_mandelbrot_vectorized(
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    mu_vec, sigma_vec = benchmark(
        compute_mandelbrot_vectorized,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    print(f"NumPy: {mu_vec:.4f} ± {sigma_vec:.4f} s")
    print(f"Speedup vs naive: {mu_naive / mu_vec:.2f}x")


    # =============================
    # NUMBA
    # =============================
    print("\nNumba version:")

    # Warm-up (compilation)
    compute_mandelbrot_numba(
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    result_numba = compute_mandelbrot_numba(
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    mu_numba, sigma_numba = benchmark(
        compute_mandelbrot_numba,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter
    )

    print(f"Numba: {mu_numba:.4f} ± {sigma_numba:.4f} s")
    print(f"Speedup vs naive: {mu_naive / mu_numba:.2f}x")
    print(f"Speedup vs NumPy: {mu_vec / mu_numba:.2f}x")


    # =============================
    # PLOTTING
    # =============================

    # Plot naive
    plt.figure(figsize=(6, 6))
    plt.imshow(result_naive, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.title("Mandelbrot (Naive)")
    plt.colorbar()
    plt.show()

    # Plot NumPy
    plt.figure(figsize=(6, 6))
    plt.imshow(result_vec, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.title("Mandelbrot (NumPy)")
    plt.colorbar()
    plt.show()

    # Plot Numba
    plt.figure(figsize=(6, 6))
    plt.imshow(result_numba, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.title("Mandelbrot (Numba)")
    plt.colorbar()
    plt.show()


    # =============================
    # PRECISION COMPARISON
    # =============================
    print("\nGenerating precision comparison plots...")

    r32 = mandelbrot_numba_f32(
        xmin, xmax, ymin, ymax,
        width, height, max_iter
    )

    r64 = mandelbrot_numba_f64(
        xmin, xmax, ymin, ymax,
        width, height, max_iter
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(r32, cmap="hot", origin="lower",
                   extent=[xmin, xmax, ymin, ymax])
    axes[0].set_title("float32")
    axes[0].axis("off")

    axes[1].imshow(r64, cmap="hot", origin="lower",
                   extent=[xmin, xmax, ymin, ymax])
    axes[1].set_title("float64")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("precision_comparison.png", dpi=150)
    plt.show()


    # =============================
    # PRECISION BENCHMARK (μ ± σ)
    # =============================
    print("\n=== Precision Benchmark ===")

    mu32, sigma32 = benchmark(
        mandelbrot_numba_f32,
        xmin, xmax, ymin, ymax,
        width, height, max_iter
    )

    mu64, sigma64 = benchmark(
        mandelbrot_numba_f64,
        xmin, xmax, ymin, ymax,
        width, height, max_iter
    )

    print("Precision    Time (μ ± σ) [s]")
    print("--------------------------------")
    print(f"float32      {mu32:.4f} ± {sigma32:.4f}")
    print(f"float64      {mu64:.4f} ± {sigma64:.4f}")

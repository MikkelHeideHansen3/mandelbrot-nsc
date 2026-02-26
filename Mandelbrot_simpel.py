"""
Mandelbrot Set Generator 
Author: Mikkel Heide Hansen
Course: Numerical Scientific Computing (NSC) 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import statistics

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

def row_sums(A):
    N = A.shape[0]
    s = 0.0
    for i in range(N):
        s += np.sum(A[i, :])
    return s


def column_sums(A):
    N = A.shape[1]
    s = 0.0
    for j in range(N):
        s += np.sum(A[:, j])
    return s

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

    # =========================
    # Validate results
    # =========================
    print("\nValidating results...")

    if np.allclose(result_naive, result_vec):
        print("Results match!")
    else:
        print("Results differ!")

    diff = np.abs(result_naive - result_vec)
    print(f"Max difference: {diff.max()}")
    print(f"Different pixels: {(diff > 0).sum()}")

    print(f"\nSpeedup: {t_naive / t_vec:.2f}x faster")
    #Side quest - 
    print("\n===== Memory Access Pattern =====")

    N = 5000   # 10000 kan være tung på 8GB RAM – start med 5000
    A = np.random.rand(N, N)

    print("\nC-order array (row-major, default NumPy):")

    print("Row sums:")
    t_row, _ = benchmark(row_sums, A, n_runs=3)

    print("Column sums:")
    t_col, _ = benchmark(column_sums, A, n_runs=3)

    print(f"\nRow/Column speed ratio: {t_col / t_row:.2f}x slower")

    print("\nNow testing Fortran-order (column-major):")
    A_f = np.asfortranarray(A)

    print("Row sums (Fortran-order):")
    t_row_f, _ = benchmark(row_sums, A_f, n_runs=3)

    print("Column sums (Fortran-order):")
    t_col_f, _ = benchmark(column_sums, A_f, n_runs=3)

    print(f"\nRow/Column speed ratio (Fortran): {t_row_f / t_col_f:.2f}x slower")

    print("\n===== Problem Size Scaling =====")

    sizes = [256, 512, 1024, 2048, 4096]
    runtimes = []

    for N in sizes:
        print(f"\nRunning size {N} x {N}")

        t, _ = benchmark(
            compute_mandelbrot_vectorized,
            xmin, xmax, ymin, ymax,
            N, N,
            max_iter,
            n_runs=3
        )

        runtimes.append(t)

    print("Sizes:", sizes)
    print("Runtimes:", runtimes)

    # Plot AFTER loop
    if len(sizes) == len(runtimes):
        plt.figure()
        plt.plot(sizes, runtimes, marker='o')
        plt.xlabel("Grid size (N x N)")
        plt.ylabel("Runtime (seconds)")
        plt.title("Mandelbrot Scaling (Vectorized)")
        plt.grid(True)
        plt.show()
    else:
        print("ERROR: sizes and runtimes length mismatch")

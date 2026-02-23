""" 
Mandelbrot Set Generator
Author: Mikkel Heide Hansen
Course: Numerical Scientific Computing (NSC) 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import statistics



def mandelbrot_point(c, max_iter):
    """
    Compute iteration count for a single complex point c.
    """
    z = 0.0 + 0.0j
    for n in range(max_iter):
        if abs(z) >= 2:
            return n
        z = z*z + c
    return max_iter

def compute_mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """
    Compute Mandelbrot set over a 2D grid.
    """
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
    """
    Compute Mandelbrot set over a 2D grid.
    """
    # Create complex grid 
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # Initialize arrays
    Z = np.zeros_like(C, dtype=np.complex128)
    M = np.zeros(C.shape, dtype=int)

    # Keep only iteration loop
    for n in range(max_iter):
        
        mask = np.abs(Z) <= 2  # points not yet escaped

        if not mask.any():
            break
        # Update only active points
        Z[mask] = Z[mask]**2 + C[mask]

        # Count iterations
        M[mask] += 1


    return M

#Memory Access Pattern function
def row_sums(A):
    N = A.shape[0]
    s = 0.0
    for i in range(N):
        s += np.sum(A[i, :])
    return s

#Memory Access Pattern function
def column_sums(A):
    N = A.shape[1]
    s = 0.0
    for j in range(N):
        s += np.sum(A[:, j])
    return s


def benchmark(func, *args, n_runs=5):
    """
    Time func and return median runtime over n_runs.
    """
    times = []

    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args)
        times.append(time.perf_counter() - t0)

    median_t = statistics.median(times)

    print(f"Median: {median_t:.4f}s "
          f"(min={min(times):.4f}, max={max(times):.4f})")

    return median_t, result



if __name__ == "__main__":
    # Parameters (typical view)
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100


        # Benchmark computation naive
    print("Naive version:")
    t_naive, result_naive = benchmark(
        compute_mandelbrot_naive,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter,
        n_runs=3
    )
        # Benchmark computation vectorized
    print("\nVectorized version:")
    t_vec, result_vec = benchmark(
        compute_mandelbrot_vectorized,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter,
        n_runs=3
    )


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


    # Visualization naive
    plt.figure(figsize=(6, 6))
    plt.imshow(result_naive, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.colorbar(label="Iterations")
    plt.title("Mandelbrot Set (Naive)")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.tight_layout()
    plt.show()

    # Visualization vector
    plt.figure(figsize=(6, 6))
    plt.imshow(result_vec, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.colorbar(label="Iterations")
    plt.title("Mandelbrot Set (Vectorized NumPy)")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.tight_layout()
    plt.show()
"""
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

# Plot scaling
plt.figure()
plt.plot(sizes, runtimes, marker='o')
plt.xlabel("Grid size (N x N)")
plt.ylabel("Runtime (seconds)")
plt.title("Mandelbrot Scaling (Vectorized)")
plt.grid(True)
plt.show()
"""
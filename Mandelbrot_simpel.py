""" 
Mandelbrot Set Generator
Author: Mikkel Heide Hansen
Course: Numerical Scientific Computing 2026
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
        z = z*z + c
        if abs(z) > 2:
            return n
    return max_iter


def compute_mandelbrot_naive(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """
    Compute Mandelbrot set over a 2D grid.
    """
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y   # ← Milestone 1

    result = np.zeros(C.shape, dtype=int)

    for i in range(height):
        for j in range(width):
            result[i, j] = mandelbrot_point(C[i, j], max_iter)

    return result

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

        # Benchmark computation
    t, mandelbrot = benchmark(
        compute_mandelbrot_naive,
        xmin, xmax, ymin, ymax,
        width, height,
        max_iter,
        n_runs=3
    )


    # Visualization
    plt.figure(figsize=(6, 6))
    plt.imshow(mandelbrot, extent=[xmin, xmax, ymin, ymax],
               origin="lower", cmap="hot")
    plt.colorbar(label="Iterations")
    plt.title("Mandelbrot Set (Naive Python)")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.tight_layout()
    plt.show()

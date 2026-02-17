"""
Mandelbrot Set Generator
Author: Mikkel Heide Hansen
Course: Numerical Scientific Computing 2026
"""
import numpy as np
import matplotlib.pyplot as plt
import time


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


def compute_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter=100):
    """
    Compute Mandelbrot set over a 2D grid.
    """
    result = np.zeros((height, width), dtype=int)

    x_vals = np.linspace(xmin, xmax, width)
    y_vals = np.linspace(ymin, ymax, height)

    for i, y in enumerate(y_vals):
        for j, x in enumerate(x_vals):
            c = x + 1j*y
            result[i, j] = mandelbrot_point(c, max_iter)

    return result


if __name__ == "__main__":
    # Parameters (typical view)
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 1024, 1024
    max_iter = 100

    # Timing
    start = time.time()
    mandelbrot = compute_mandelbrot(
        xmin, xmax, ymin, ymax, width, height, max_iter
    )
    elapsed = time.time() - start
    print(f"Computation took {elapsed:.3f} seconds")

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

import numpy as np
from line_profiler import profile

@profile
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


if __name__ == "__main__":
    compute_mandelbrot_vectorized(-2, 1, -1.5, 1.5, 512, 512, 100)
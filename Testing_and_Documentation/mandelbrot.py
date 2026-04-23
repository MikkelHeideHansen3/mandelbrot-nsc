import numpy as np
from numba import njit
from multiprocessing import Pool
from dask import delayed
import dask


@njit(cache=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """
    Compute the escape iteration count for a single Mandelbrot point.

    Parameters
    ----------
    c_real : float
        Real part of the complex number c.
    c_imag : float
        Imaginary part of the complex number c.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    int
        Iteration at which the sequence escapes, or max_iter if it does not.
    """
    zr = 0.0
    zi = 0.0

    for i in range(max_iter):
        zr2 = zr * zr
        zi2 = zi * zi

        if zr2 + zi2 > 4.0:
            return i

        zi = 2.0 * zr * zi + c_imag
        zr = zr2 - zi2 + c_real

    return max_iter


@njit(cache=True)
def mandelbrot_chunk(
    row_start: int,
    row_end: int,
    N: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int
) -> np.ndarray:
    """
    Compute a chunk (subset of rows) of the Mandelbrot grid.

    Parameters
    ----------
    row_start : int
        Starting row index.
    row_end : int
        Ending row index (exclusive).
    N : int
        Grid size (NxN).
    xmin, xmax : float
        Bounds of the real axis.
    ymin, ymax : float
        Bounds of the imaginary axis.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        2D array of escape iteration counts for the chunk.
    """
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


def mandelbrot_serial(
    N: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int = 100
) -> np.ndarray:
    """
    Compute the Mandelbrot set using a serial implementation.

    Parameters
    ----------
    N : int
        Grid size (NxN).
    xmin, xmax : float
        Bounds of the real axis.
    ymin, ymax : float
        Bounds of the imaginary axis.
    max_iter : int, optional
        Maximum number of iterations (default is 100).

    Returns
    -------
    np.ndarray
        2D array of escape iteration counts.
    """
    return mandelbrot_chunk(
        0, N,
        N,
        xmin, xmax,
        ymin, ymax,
        max_iter
    )


def _worker(args: tuple) -> np.ndarray:
    """
    Worker function for multiprocessing.

    Parameters
    ----------
    args : tuple
        Arguments for mandelbrot_chunk.

    Returns
    -------
    np.ndarray
        Computed chunk of the Mandelbrot grid.
    """
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(
    N: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int = 100,
    n_workers: int = 2
) -> np.ndarray:
    """
    Compute the Mandelbrot set using multiprocessing.

    Parameters
    ----------
    N : int
        Grid size (NxN).
    xmin, xmax : float
        Bounds of the real axis.
    ymin, ymax : float
        Bounds of the imaginary axis.
    max_iter : int, optional
        Maximum number of iterations.
    n_workers : int, optional
        Number of worker processes.

    Returns
    -------
    np.ndarray
        2D array of escape iteration counts.
    """
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
        parts = pool.map(_worker, chunks)

    return np.vstack(parts)


def mandelbrot_dask(
    N: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int = 100,
    n_chunks: int = 4
) -> np.ndarray:
    """
    Compute the Mandelbrot set using Dask parallelisation.

    Parameters
    ----------
    N : int
        Grid size (NxN).
    xmin, xmax : float
        Bounds of the real axis.
    ymin, ymax : float
        Bounds of the imaginary axis.
    max_iter : int, optional
        Maximum number of iterations.
    n_chunks : int, optional
        Number of chunks for parallel computation.

    Returns
    -------
    np.ndarray
        2D array of escape iteration counts.
    """
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

    tasks = [delayed(mandelbrot_chunk)(*chunk) for chunk in chunks]
    parts = dask.compute(*tasks)

    return np.vstack(parts)
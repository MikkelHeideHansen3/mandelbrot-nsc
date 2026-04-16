import numpy as np
import pytest
from numba import njit
from multiprocessing import Pool
from dask import delayed
import dask

# =========================
# Functions
# =========================

@njit(cache=True)
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


@njit(cache=True)
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


def mandelbrot_serial(N, xmin, xmax, ymin, ymax, max_iter=100):
    return mandelbrot_chunk(
        0, N,
        N,
        xmin, xmax,
        ymin, ymax,
        max_iter
    )


def _worker(args):
    return mandelbrot_chunk(*args)


def mandelbrot_parallel(N, xmin, xmax, ymin, ymax,
                        max_iter=100, n_workers=2):

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


def mandelbrot_dask(N, xmin, xmax, ymin, ymax,
                    max_iter=100, n_chunks=4):

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


# =========================
# Helper Functions
# =========================

def pixel(c, max_iter):
    return mandelbrot_pixel(c.real, c.imag, max_iter)


# =========================
# Tests
# =========================

# --- Test 1: Known analytical values (parametrize)
CASES = [
    (0+0j, 100, 100),   # inside set
    (5+0j, 100, 1),     # far outside
    (-2.5+0j, 100, 1),  # left side
]

@pytest.mark.parametrize("c, max_iter, expected", CASES)
def test_pixel_known_values(c, max_iter, expected):
    assert pixel(c, max_iter) == expected


# --- Test 2: Result is always in valid range
def test_pixel_range():
    for c in [0+0j, 1+1j, -2+0j]:
        result = pixel(c, 100)
        assert 0 <= result <= 100


# --- Test 3: Multiprocessing matches serial
def test_parallel_matches_serial():
    N = 32
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    serial = mandelbrot_serial(N, xmin, xmax, ymin, ymax, 50)
    parallel = mandelbrot_parallel(N, xmin, xmax, ymin, ymax, 50)

    assert np.array_equal(serial, parallel)


# --- Test 4: Dask matches serial
def test_dask_matches_serial():
    N = 32
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5

    serial = mandelbrot_serial(N, xmin, xmax, ymin, ymax, 50)
    dask_res = mandelbrot_dask(N, xmin, xmax, ymin, ymax, 50, n_chunks=4)

    assert np.array_equal(serial, dask_res)
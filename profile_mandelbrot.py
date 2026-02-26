from Mandelbrot_simpel import (
    compute_mandelbrot_naive,
    compute_mandelbrot_vectorized
)

import cProfile
import pstats


def profile_function(func, name):
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 512, 512
    max_iter = 100

    profiler = cProfile.Profile()
    profiler.enable()

    func(xmin, xmax, ymin, ymax, width, height, max_iter)

    profiler.disable()

    print(f"\n===== PROFILE: {name} =====")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(15)


if __name__ == "__main__":
    profile_function(compute_mandelbrot_naive, "NAIVE")
    profile_function(compute_mandelbrot_vectorized, "NUMPY")
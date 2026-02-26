from Mandelbrot_simpel import compute_mandelbrot_naive
import cProfile
import pstats
def run_profiles():
    xmin, xmax = -2.0, 1.0
    ymin, ymax = -1.5, 1.5
    width, height = 512, 512
    max_iter = 100

    profiler = cProfile.Profile()
    profiler.enable()
    compute_mandelbrot_naive(
        xmin, xmax, ymin, ymax,
        width, height, max_iter
    )
    profiler.disable()

    print("\n===== NAIVE PROFILE =====")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(15)

if __name__ == "__main__":
    run_profiles()
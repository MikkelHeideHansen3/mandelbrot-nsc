import cProfile
import pstats

from Mandelbrot_simpel import compute_mandelbrot_naive, compute_mandelbrot_vectorized


# Profile naive
cProfile.run(
    "compute_mandelbrot_naive(-2, 1, -1.5, 1.5, 512, 512, 100)",
    "naive_profile.prof"
)

# Profile numpy
cProfile.run(
    "compute_mandelbrot_vectorized(-2, 1, -1.5, 1.5, 512, 512, 100)",
    "numpy_profile.prof"
)


# Print results
for name in ["naive_profile.prof", "numpy_profile.prof"]:
    print("\n======", name, "======")
    stats = pstats.Stats(name)
    stats.sort_stats("cumulative")
    stats.print_stats(10)
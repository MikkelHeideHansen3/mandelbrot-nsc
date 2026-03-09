import math
import random
import time
import statistics


def estimate_pi_serial(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x = random.random()
        y = random.random()

        if x*x + y*y <= 1:
            inside_circle += 1

    return 4 * inside_circle / num_samples


if __name__ == "__main__":
    num_samples = 10_000_000
    times = []

    for _ in range(3):
        t0 = time.perf_counter()

        pi_estimate = estimate_pi_serial(num_samples)

        t1 = time.perf_counter()
        times.append(t1 - t0)

    t_serial = statistics.median(times)

    print(f"pi estimate: {pi_estimate:.6f}")
    print(f"error: {abs(pi_estimate - math.pi):.6f}")
    print(f"serial time: {t_serial:.3f} s")
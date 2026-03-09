import os
import random
import time
import statistics
from multiprocessing import Pool


def estimate_pi_chunk(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        x = random.random()
        y = random.random()

        if x*x + y*y <= 1:
            inside_circle += 1

    return inside_circle


def estimate_pi_parallel(num_samples, num_processes):
    samples_per_process = num_samples // num_processes
    tasks = [samples_per_process] * num_processes

    with Pool(processes=num_processes) as pool:
        results = pool.map(estimate_pi_chunk, tasks)

    total_inside = sum(results)

    return 4 * total_inside / num_samples


if __name__ == "__main__":

    num_samples = 10_000_000

    for num_proc in range(1, os.cpu_count() + 1):

        times = []

        for _ in range(3):
            t0 = time.perf_counter()

            pi_est = estimate_pi_parallel(num_samples, num_proc)

            times.append(time.perf_counter() - t0)

        t = statistics.median(times)

        print(f"{num_proc:2d} workers: {t:.3f}s   pi={pi_est:.6f}")
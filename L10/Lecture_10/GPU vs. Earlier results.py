import matplotlib.pyplot as plt

labels_1024 = [
    "Naive",
    "NumPy",
    "Numba f32",
    "Numba f64",
    "GPU f32",
    "GPU f64"
]

times_1024 = [
    6.808,
    0.812,
    0.0802,
    0.0800,
    0.0386,
    0.0457
]

plt.figure(figsize=(8,4))
plt.bar(labels_1024, times_1024)

plt.yscale("log")
plt.ylabel("Runtime [s] (log scale)")
plt.title("Mandelbrot Performance (1024×1024)")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

labels_4096 = [
    "Numba f32",
    "Numba f64",
    "Multiprocessing",
    "Dask local",
    "Dask cluster",
    "GPU f32",
    "GPU f64"
]

times_4096 = [
    1.2607,
    1.2602,
    0.580,
    0.565,
    0.460,
    0.388,
    0.416
]

plt.figure(figsize=(10,5))
plt.bar(labels_4096, times_4096)

plt.yscale("log")
plt.ylabel("Runtime [s] (log scale)")
plt.title("Mandelbrot Performance (4096×4096)")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
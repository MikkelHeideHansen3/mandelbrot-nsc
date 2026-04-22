import pyopencl as cl
import numpy as np
import time
import matplotlib.pyplot as plt

# -----------------------------
# KERNELS
# -----------------------------

# --- float32 ---
KERNEL_SRC_F32 = """
__kernel void mandelbrot_f32(
    __global int *result,
    const float x_min, const float x_max,
    const float y_min, const float y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) return;

    float c_real = x_min + col * (x_max - x_min) / (float)N;
    float c_imag = y_min + row * (y_max - y_min) / (float)N;

    float zr = 0.0f;
    float zi = 0.0f;
    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0f) {
        float tmp = zr*zr - zi*zi + c_real;
        zi = 2.0f * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""

# --- float64 ---
KERNEL_SRC_F64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void mandelbrot_f64(
    __global int *result,
    const double x_min, const double x_max,
    const double y_min, const double y_max,
    const int N, const int max_iter)
{
    int col = get_global_id(0);
    int row = get_global_id(1);

    if (col >= N || row >= N) return;

    double c_real = x_min + col * (x_max - x_min) / (double)N;
    double c_imag = y_min + row * (y_max - y_min) / (double)N;

    double zr = 0.0;
    double zi = 0.0;
    int count = 0;

    while (count < max_iter && zr*zr + zi*zi <= 4.0) {
        double tmp = zr*zr - zi*zi + c_real;
        zi = 2.0 * zr * zi + c_imag;
        zr = tmp;
        count++;
    }

    result[row * N + col] = count;
}
"""

# -----------------------------
# SETUP
# -----------------------------
ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

prog_f32 = cl.Program(ctx, KERNEL_SRC_F32).build()
prog_f64 = cl.Program(ctx, KERNEL_SRC_F64).build()

# -----------------------------
# PARAMETERS
# -----------------------------
N = 2048
MAX_ITER = 200

X_MIN, X_MAX = -2.5, 1.0
Y_MIN, Y_MAX = -1.25, 1.25

image = np.zeros((N, N), dtype=np.int32)
image_dev = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, image.nbytes)

# -----------------------------
# WARM-UP (important)
# -----------------------------
prog_f32.mandelbrot_f32(
    queue, (64, 64), None,
    image_dev,
    np.float32(X_MIN), np.float32(X_MAX),
    np.float32(Y_MIN), np.float32(Y_MAX),
    np.int32(64), np.int32(MAX_ITER)
)
queue.finish()

# -----------------------------
# TIME F32
# -----------------------------
t0 = time.perf_counter()

prog_f32.mandelbrot_f32(
    queue, (N, N), None,
    image_dev,
    np.float32(X_MIN), np.float32(X_MAX),
    np.float32(Y_MIN), np.float32(Y_MAX),
    np.int32(N), np.int32(MAX_ITER)
)

queue.finish()
t_f32 = time.perf_counter() - t0

# -----------------------------
# TIME F64
# -----------------------------
# warm-up f64
prog_f64.mandelbrot_f64(
    queue, (64, 64), None,
    image_dev,
    np.float64(X_MIN), np.float64(X_MAX),
    np.float64(Y_MIN), np.float64(Y_MAX),
    np.int32(64), np.int32(MAX_ITER)
)

t0 = time.perf_counter()


prog_f64.mandelbrot_f64(
    queue, (N, N), None,
    image_dev,
    np.float64(X_MIN), np.float64(X_MAX),
    np.float64(Y_MIN), np.float64(Y_MAX),
    np.int32(N), np.int32(MAX_ITER)
)

queue.finish()
t_f64 = time.perf_counter() - t0

# -----------------------------
# COPY RESULT (from last run)
# -----------------------------
cl.enqueue_copy(queue, image, image_dev)
queue.finish()

# -----------------------------
# RESULTS
# -----------------------------
print(f"f32 time: {t_f32*1e3:.1f} ms")
print(f"f64 time: {t_f64*1e3:.1f} ms")
print(f"ratio (f64/f32): {t_f64/t_f32:.2f}")

# -----------------------------
# PLOT
# -----------------------------
plt.imshow(image, cmap="hot", origin="lower")
plt.axis("off")
plt.title("Mandelbrot (last run = f64)")
plt.show()
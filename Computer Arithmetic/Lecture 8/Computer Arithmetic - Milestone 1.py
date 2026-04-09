import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 512
MAX_ITER = 1000
TAU = 0.01

# Region (Seahorse Valley)
x = np.linspace(-0.7530, -0.7490, N)
y = np.linspace(0.0990, 0.1030, N)

# Build complex grids
C64 = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).astype(np.complex128)
C32 = C64.astype(np.complex64)

# Initialize
z32 = np.zeros_like(C32)
z64 = np.zeros_like(C64)

diverge = np.full((N, N), MAX_ITER, dtype=np.int32)
active = np.ones((N, N), dtype=bool)

# Iterate
for k in range(MAX_ITER):
    if not active.any():
        break

    z32[active] = z32[active]**2 + C32[active]
    z64[active] = z64[active]**2 + C64[active]

    # Difference between float32 and float64
    diff = (
        np.abs(z32.real.astype(np.float64) - z64.real)
        + np.abs(z32.imag.astype(np.float64) - z64.imag)
    )

    newly = active & (diff > TAU)

    diverge[newly] = k
    active[newly] = False

# Plot
plt.imshow(
    diverge,
    cmap="plasma",
    origin="lower",
    extent=[-0.7530, -0.7490, 0.0990, 0.1030],
)
plt.colorbar(label="First divergence iteration")
plt.title(f"Trajectory divergence (tau={TAU})")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.show()

fraction = np.sum(diverge < MAX_ITER) / (N * N)
print(f"Fraction of pixels that diverge: {fraction:.4f}")
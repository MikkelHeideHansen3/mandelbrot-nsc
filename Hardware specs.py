import platform
import psutil
import numpy as np
import numba

print("CPU:", platform.processor())
print("Cores:", psutil.cpu_count(logical=True))
print("RAM (GB):", round(psutil.virtual_memory().total/1e9,2))
print("NumPy:", np.__version__)
print("Numba:", numba.__version__)
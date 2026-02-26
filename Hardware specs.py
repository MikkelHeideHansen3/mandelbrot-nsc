import platform
import psutil
import numpy as np
import numba
import sys
import matplotlib

print("CPU:", platform.processor())
print("Cores:", psutil.cpu_count(logical=True))
print("RAM (GB):", round(psutil.virtual_memory().total/1e9,2))
print("NumPy:", np.__version__)
print("Numba:", numba.__version__)
print(f"Python:      {sys.version.split()[0]}")
print(f"Matplotlib:  {matplotlib.__version__}")
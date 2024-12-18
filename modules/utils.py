# --------------------------------------------
# File: modules/utils.py
# --------------------------------------------
import psutil
import os

def set_single_core_affinity() -> None:
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity([0])
    except (AttributeError, psutil.AccessDenied, NotImplementedError):
        print("Warning: Setting CPU affinity is not supported on this platform or access is denied.")

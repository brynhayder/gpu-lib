"""Tools for NVIDIA GPUs"""
from functools import wraps
import time
import warnings

from py3nvml import py3nvml


class NVMLSession:
    def __init__(self):
        self._initialised = False

    def __bool__(self):
        return bool(self._initialised)

    def __enter__(self):
        try:
            py3nvml.nvmlInit()
            self._initialised = True
        except py3nvml.NVMLError_LibraryNotFound:
            self._initialised = False
            warnings.warn("No GPUs found")
            return None
        else:
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._initialised:
            py3nvml.nvmlShutdown()
        if exc_type is not None:
            raise


def _is_free(gpu_handle):
    try:
        procs = py3nvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)
    except py3nvml.NVMLError as e:
        warnings.warn(f"Error when accessing processes of handle {gpu_handle}.")
        return False
    else:
        return not procs


def free_gpus(ignore=None):
    ignore = ignore or []
    free = list()

    with NVMLSession() as s:
        if not s:
            return None
        num_gpus = py3nvml.nvmlDeviceGetCount()
        
        for i in range(num_gpus):
            if i in ignore:
                continue
            handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
            if _is_free(handle):
                free.append(i)
    return free


# This wrapper idea is cool but it doesn't work with cuda visible devices
# since that needs to be set before torch is imported. 
# Also 
class ExecWhenFree:
    """UNTESTED wrapper to exec a function when gpus are free
    
    wrapped function must expect gpu_id list as first argument
    """
    def __init__(self, n_gpus, timeout=None, wait=1, ignore=None):
        self.n_gpus = n_gpus
        self.timeout = timeout or float('inf')
        self.wait = wait
        self.ignore = ignore or []
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            free = free_gpus(ignore=self.ignore)
            if len(free) >= self.n_gpus:
                return func(free, *args, **kwargs)
            else:
                elapsed = time.time() - start
                if elapsed > self.timeout: 
                    # Will need to sort this logging out
                    # logging.info("Timeout exceeded")
                    raise TimeoutError("Timeout exceeded")
                time.sleep(self.wait)
        return wrapper



"""
Notes:
    - I want to write unittests and more testable code in general
    - At the start is always a good time to write tests! Need to read up on TDD

TODO:
    - Write a class that, given commands, waits for free GPUs and then dispatches on them
    - Maybe a good way to do this is to just have a decorator that decorates a main function...?

"""

if __name__ == "__main__":

    print("Free GPUs")
    print(free_gpus())


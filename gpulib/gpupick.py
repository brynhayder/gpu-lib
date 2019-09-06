"""Tools for NVIDIA GPUs"""
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


def free_gpus(ignore=None):
    ignore = ignore or []
    
    with NVMLSession() as s:
        if not s:
            return None
         
     
    return [k for k in free if k not in ignore]



if __name__ == "__main__":

    print("Free GPUs")
    print(free_gpus())


"""Tools for NVIDIA GPUs"""

import py3nvml


def free_gpus(ignore=None):
    ignore = ignore or []
    py3nvml.nvmlInit()
     
    return [k for k in free if k not in ignore]


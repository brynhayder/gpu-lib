import argparse
import os
import subprocess
import time

from py3nvml import py3nvml

from .utils import free_gpus, NVMLSession


class GPUNotFoundError(Exception):
    pass


def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--n', '-n',
            type=int,
            default=1,
            help='Number o  GPUs (default: %(default)s)'
        )
   
    parser.add_argument(
            '--timeout', '-t',
            type=float,
            default=float('inf'),
            help='How long to wait in seconds for GPUs before exiting. Default: %(default)f.'
        )

    parser.add_argument(
            '--wait', '-w',
            type=float,
            default=1,
            help="How long to wait in seconds between checking for free gpus. Default: %(default)f."
        )

    parser.add_argument('command', nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_cmd()
    if not args.command:
        raise ValueError("No command given.")

    with NVMLSession() as sess:
        if not sess:
            raise GPUNotFoundError("No GPUs found.")
        
        n_devices = py3nvml.nvmlDeviceGetCount()
        if args.n > n_devices:
            raise ValueError(f"Requested devices {args.n} > {n_devices} devices found.")
    
    print(f"Waiting for {args.n} free GPUs")
    start = time.time()
    elapsed = 0
    while elapsed < args.timeout:
        free = free_gpus()
        if len(free) >= args.n:
            print(f"Running command: {' '.join(args.command)}")
            to_use = free[:args.n]
            print(f"On gpus: {to_use}")
            os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, to_use))
            subprocess.run(args.command)
            break
        time.sleep(args.wait)
        elapsed = time.time() - start
    else:
        raise TimeoutError("Timeout exceeded")
    return 0


if __name__ == "__main__":
    main()


import datetime
import logging
import os
import socket
import subprocess
import time
from types import SimpleNamespace

import torch

from xllmx.util.misc import random_seed

logger = logging.getLogger(__name__)


def find_free_port(start_port: int, end_port: int):
    """
    Find a free port within the specified range.
    """
    for port in range(start_port, end_port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", port))  # Try to bind to the port
            s.close()  # Close the socket if successful
            return port
        except OSError as e:
            continue
    raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")


def init_distributed_mode(args=SimpleNamespace()):
    random_seed(getattr(args, "seed", 0))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.local_rank = args.gpu
        args.dist_url = "env://"
    elif "SLURM_PROCID" in os.environ:
        os.environ["MASTER_PORT"] = "6891"
        while "MASTER_ADDR" not in os.environ or len(os.environ["MASTER_ADDR"].strip()) == 0:
            os.environ["MASTER_ADDR"] = (
                subprocess.check_output(
                    "sinfo -Nh -n %s | head -n 1 | awk '{print $1}'" % os.environ["SLURM_NODELIST"],
                    shell=True,
                )
                .decode()
                .strip()
            )
            time.sleep(1)
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        args.local_rank = args.gpu
        args.dist_url = "env://"
        os.environ["LOCAL_RANK"] = str(args.gpu)
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["RANK"] = str(args.rank)
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port(9000, 10000))
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        args.rank = 0
        args.gpu = args.local_rank = 0
        args.world_size = 1
        args.dist_url = "env://"

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}, gpu {}".format(args.rank, args.dist_url, args.gpu), flush=True)
    # Do NOT pass device_id here: on PyTorch 2.x it triggers eager NCCL connect
    # (eager_connect_single_device) which fails hard on misconfigured IB/RoCE.
    # Bind the CUDA device first, init PG, then barrier with explicit device_ids.
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(seconds=2 * 60 * 60),
    )
    # Skip NCCL barrier here: init_process_group already syncs via the TCP store.
    # An early NCCL barrier is fragile on multi-node (and masks rank collisions as
    # socketStartConnect aborts). First real collective happens later in training.

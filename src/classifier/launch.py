#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from argparse import ArgumentParser, REMAINDER
import os
import re
import socket
import subprocess
import sys

import GPUtil


r"""
`torch.distributed.launch` is a module that spawns up multiple distributed
training processes on each of the training nodes.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be benefitial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc_per_node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    >>> python -m torch.distributed.launch --help


**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(arg.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(arg.local_rank):
    >>>    # your code to run

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.

::

    torch.distributed.init_process_group(backend='YOUR BACKEND')

4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.

::

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.local_rank],
                                                      output_device=arg.local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility

5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use_env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.

"""



num_gpus_here = len(GPUtil.getGPUs())

def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="PyTorch distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")

    # Optional arguments for the launch helper

    parser.add_argument("--node_rank", type=int, default=0,
                        help="this machine's index into the number of machines (0 is the master);"
                             "default: 0"
                              )
    parser.add_argument("--nother_gpus", type=int, default=0,
                        help="Total number of GPUs used on other nodes; default: 0",
                              )
    parser.add_argument("--nhere_gpus", type=int,
                        help=f"number of GPUs to use on this node; default is all: {num_gpus_here}",
                        default=num_gpus_here
                        )
    parser.add_argument("--master_addr", default="127.0.0.1", type=str,
                        help="Master node (rank 0)'s address, should be either "
                             "the IP address or the hostname of node 0, for "
                             "single node multi-proc training, the "
                             "--master_addr can simply be 127.0.0.1")
    parser.add_argument("--master_port", default=29500, type=int,
                        help="Master node (rank 0)'s free port that needs to "
                             "be used for communication during distributed "
                             "training")
    parser.add_argument("-m", "--module", default=False, action="store_true",
                        help="Changes each process to interpret the launch script "
                             "as a python module, executing with the same behavior as"
                             "'python -m'.")
    parser.add_argument("--no_python", default=False, action="store_true",
                        help="Do not prepend the training script with \"python\" - just exec "
                             "it directly. Useful when the script is not a Python script, "
                             "or has a #! at the top."
                        )

    # positional
    parser.add_argument("training_script", type=str,
                        help="The full path to the single GPU training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")

    # Rest of args are for the training program:
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    args = parse_args()

    # world size in terms of number of processes, which is
    # equal to number of GPUs used on all machines together.
    # Number of GPUs used on the other nodes, plus the ones
    # used on this machine:
    
    dist_world_size = args.nother_gpus + args.nhere_gpus
    #dist_world_size = args.nproc_per_node * args.nnodes

    # If host name was provided instead of an
    # IP address, resolve that:
    if re.search('^[\d.]*$', args.master_addr) is None:
        # Got not an IP address, but something
        # with letters:
        master_addr = socket.gethostbyname(args.master_addr)
    else:
        master_addr = args.master_addr 

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    
    processes = []

    if 'OMP_NUM_THREADS' not in os.environ and args.nhere_gpus > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print("*****************************************\n"
              "Setting OMP_NUM_THREADS environment variable for each process "
              "to be {} in default, to avoid your system being overloaded, "
              "please further tune the variable for optimal performance in "
              "your application as needed. \n"
              "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    # Compute a unique number for each GPU within
    # the group of nodes (machines). Starting with
    # the master node, whose numbers are 0,1,...<ngpus_here>:
    
    for local_rank in range(0, args.nhere_gpus):

        # To ensure no dups in the global GPU ids, 
        # assume every node has as many GPUs as the
        # all nodes taken together:
        
        dist_rank = dist_world_size * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag and the '--module' flag at the same time.")

        cmd.append(args.training_script)

        # To the args for the train script processes,
        # add --started_from_launch to let each of
        # them know:
        args_for_train_scripts = ['--started_from_launch'] + \
                                 args.training_script_args
        cmd.extend(args_for_train_scripts)

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    #***********
    print(f"********Num processes launched: {len(processes)}")
    #***********        
    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode,
                                                cmd=cmd)


if __name__ == "__main__":
    #**********
    # try:
    #   os.remove('/tmp/epoch0.output')
    # except FileNotFoundError:
    #   pass
    # try:
    #     os.remove('/tmp/epoch1.output')
    # except FileNotFoundError:
    #   pass
    #**********  
    main()

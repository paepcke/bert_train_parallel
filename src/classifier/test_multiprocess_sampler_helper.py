#!/usr/bin/env python3

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torch import cuda

import torch.distributed as dist

import os 
torch.distributed.init_process_group(backend="nccl")
from bert_feeder_dataloader import MultiprocessingDataloader
from bert_feeder_dataset import SqliteDataset

input_size = 5
output_size = 2
batch_size = 2
data_size = 16

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

label_mapping = {0, 'right',
                 1, 'left',
                 2, 'neutral'
    }
test_db_path = os.path.join(os.path.dirname(__file__), 'datasets/test_db.sqlite')

dataset = SqliteDataset(
    'fake/csv/path',
    label_mapping,
    sqlite_path=test_db_path,
    sequence_len=128,
    text_col_name='foo',
    label_col_name='bar',
    delete_db=False,
    quiet=True
    )        

world_size = int(os.environ['WORLD_SIZE'])
node_rank  = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])

sampler = DistributedSampler(
   dataset,
   num_replicas=world_size,
   rank=node_rank
   )

dataloader = MultiprocessingDataloader(dataset, world_size, node_rank)

epoch = 0
while epoch < 2:
    t = 0
    dataloader.set_epoch(epoch) # this is key, if this line removed, the data will be same,
                                # else the data won't be same between iteration but will
                                # be same in every per run
    with open('/tmp/epoch0.output', 'a') as epoch0_fd:
        with open('/tmp/epoch1.output', 'a') as epoch1_fd:
            for data in dataloader:
                print(f"Node{node_rank} GPU{local_rank} Epoch{epoch}: {int(data['tok_ids'])}")
                if epoch == 0:
                  epoch0_fd.write(f"Node{node_rank} GPU{local_rank} {int(data['tok_ids'])}\n")
                else:
                  epoch1_fd.write(f"Node{node_rank} GPU{local_rank} {int(data['tok_ids'])}\n")
    epoch+=1

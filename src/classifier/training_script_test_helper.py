'''
Created on Jun 27, 2020

@author: paepcke
'''

import sys, os

sys.path.append(os.path.dirname(__file__))

from bert_feeder_dataset import SqliteDataset
from bert_feeder_dataloader import MultiprocessingDataloader

import torch

class TrainProcessTestHelper(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        world_size = int(os.environ['WORLD_SIZE'])
        node_rank  = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])

        torch.distributed.init_process_group(backend="nccl")

        self.compute_facilities = {0: 3, # node 0: 3 GPUs
                                   1: 3, # node 1: 2 GPUs
                                   }
        self.accumulated_data = {'epoch0' : [],
                                 'epoch1' : []
                                 }
        label_mapping = {0, 'right',
                         1, 'left',
                         2, 'neutral'
            }

        test_db_path = os.path.join(os.path.dirname(__file__), 
                                    'datasets/test_db.sqlite')

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
        
        self.dataloader = MultiprocessingDataloader(dataset, 
                                                    world_size,
                                                    node_rank=node_rank
                                                    )
        
        for epoch in range(2):
            self.run(epoch)

        print(self.accumulated_data)
        
    def run(self, epoch):
        for data in self.dataloader:
            if epoch == 0:
                # Even the sample_id comes back as
                # a tensor. Turn into an integer:
                self.accumulated_data['epoch0'].append(int(data['sample_id']))
            elif epoch == 1:
                self.accumulated_data['epoch1'].append(int(data['sample_id']))
            else:
                raise ValueError("Bad epoch")

# ------------------ Main --------------

if __name__ == '__main__':
    TrainProcessTestHelper()

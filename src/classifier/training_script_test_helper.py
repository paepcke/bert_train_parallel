'''
Created on Jun 27, 2020

@author: paepcke
'''

import sys, os

sys.path.append(os.path.dirname(__file__))

from bert_feeder_dataset import SqliteDataset
from bert_feeder_dataloader import MultiprocessingDataloader

import torch
from torch import cuda

class TrainProcessTestHelper(object):
    '''
    Pretends to be a training script that is
    forked into (potentially) multiple processes
    on multiple machines. 
    
    This minimal test script is used with 
    test_multiprocess_sampler.py. It merely draws
    samples from an Sqlite test database via a 
    distributed sampler. Each forked instance of
    this script runs through two epochs over the 
    database. 
    
    It writes the samples it draws in each epoch to 
    file, which is different for each process. The 
    main unittest (test_multiprocess_sampler.py) then 
    checks that taken together, the samples each process
    draws satisfy the following:

        o Within one epoch all processes together 
          draw exactly the samples 0, 1,2,3,...23.
        o This sequence is permuted differently in the
          two epochs.  
    
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 result_file_template
                 ):
        '''
        Constructor
        '''
        # The launch script (launch.py) will have
        # set the following OS env vars:
        
        world_size = int(os.environ['WORLD_SIZE'])
        node_rank  = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Creates an OS level process group of all
        # processes that run this script. The nccl
        # communicates among the processes, including
        # those on other machines. This communication 
        # in normal operation only occurs during back
        # prop to communicate new model parameters. 
        # In this test no communication takes place.
        torch.distributed.init_process_group(backend="nccl")

        # This process only interacts with one GPU:
        cuda.set_device(local_rank)

        # A mapping of integers to labels.
        # Used by the SqliteDataset while reading
        # .csv files to translate from the string
        # labels there to internal ints that are
        # required for training:
         
        label_mapping = {0, 'right',
                         1, 'left',
                         2, 'neutral'
            }

        # We bypass reading a CSV file, using
        # instead the already prepared test Sqlite
        # db that would have been created in an earlier
        # CSV reading:
        
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

        # Client level interaction with the dataset:
        # a torch.DataLoader using a distributed sampler.
        # world_size is the number of GPUs used across
        # all nodes (a.k.a. machines). The node_rank is
        # the number of this machine in the sequence of
        # machines whose GPUs are involved what would 
        # be distributed training. The master node has
        # node_rank 0.

        self.dataloader = MultiprocessingDataloader(dataset, 
                                                    world_size,
                                                    node_rank=node_rank
                                                    )
        
        for epoch in range(2):
            
            # *MUST* set the distributed dataloader's
            # epoch variable through the dataloader.
            # That number is used by the sampler in
            # all forks as a random seed.
            
            self.dataloader.set_epoch(epoch)
            # Get {'epoch0': [sample10, sample3, ...],
            #      'epoch1': [sample21, sample12, ...]
            #     }
            samples = self.run(epoch)
            

        #print(self.accumulated_data)

    #------------------------------------
    # run 
    #-------------------

    def run(self, epoch):
        '''
        Ask for all the samples in a loop
        Write the result to a file as a dict
        
        @param epoch:
        @type epoch:
        '''

        # Place for this process to collect the
        # samples drawn. Each process only draws
        # *some* of the samples. Only the samples
        # of all forks together will be a complete
        # set of samples from the db:
                
        accumulated_data = {'epoch0' : [],
                            'epoch1' : []
                            }

        for data in self.dataloader:
            if epoch == 0:
                # Even the sample_id comes back as
                # a tensor. Turn into an integer:
                accumulated_data['epoch0'].append(int(data['sample_id']))
            elif epoch == 1:
                accumulated_data['epoch1'].append(int(data['sample_id']))
            else:
                raise ValueError("Bad epoch")
            
        return accumulated_data

    def check_sampling_correctness(self, accumulated_data):

# ------------------ Main --------------

if __name__ == '__main__':
    TrainProcessTestHelper()

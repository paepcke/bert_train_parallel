#!/usr/bin/env python3
'''
Created on Jun 23, 2020

@author: paepcke
'''
import numpy as np

import os
import sqlite3
import unittest
import GPUtil
import torch
from torch import cuda
from itertools import accumulate

#*******
# The following is a workaround for some library
# setting the env var to 'Intel' and causing
# an error.
os.environ['MKL_THREADING_LAYER'] = 'gnu'
#*******
from bert_feeder_dataloader import MultiprocessingDataloader
from bert_feeder_dataset import SqliteDataset

#******TEST_ALL = True
TEST_ALL = False

class MultiProcessSamperTester(unittest.TestCase):
    
    test_db_path = os.path.join(os.path.dirname(__file__), 'datasets/test_db.sqlite')
    launch_script_path = os.path.join(os.path.dirname(__file__), 'launch.py')
    runtime_script     = os.path.join(os.path.dirname(__file__), 'test_multiprocess_sampler_helper.py')

    #------------------------------------
    # setupClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        super(MultiProcessSamperTester, cls).setUpClass()
        
        cls.num_cuda_devs = len(GPUtil.getGPUs())
        
        # Create a test Sqlite db:
        try:
            os.remove(cls.test_db_path)
        except FileNotFoundError:
            pass
        cls.db = sqlite3.connect(cls.test_db_path)
        cls.db.row_factory = sqlite3.Row
        
        cls.db.execute('''DROP TABLE IF EXISTS Samples''')
        cls.db.execute('''CREATE TABLE Samples (sample_id int,
                                                tok_ids varchar(50),
                                                attention_mask varchar(50),
                                                label int
                                                )

                                               ''')
        # 20 samples:
        for serial_num in range(24):
            cls.db.execute(f'''INSERT INTO Samples
                            VALUES({serial_num},
                                   "[{serial_num+1}]",
                                   "[0,1,0,0]",
                                  {serial_num+1}
                                  )
                            '''
            )

        cls.db.commit()
        
        # Setup env vars that the launch.py script 
        # would normally set up:
        
        cls.num_gpus = len(GPUtil.getGPUs())
        
        os.environ['WORLD_SIZE'] = f"{cls.num_gpus}"
        os.environ['RANK']       = '0'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

        if cls.num_gpus > 0:
            torch.distributed.init_process_group(backend="nccl")

    #------------------------------------
    # setUp 
    #-------------------
    
    def setUp(self):
        unittest.TestCase.setUp(self)

        self.compute_facilities = {0: 3, # node 0: 3 GPUs
                                   1: 3, # node 1: 2 GPUs
                                   }
        self.label_mapping = {0, 'right',
                              1, 'left',
                              2, 'neutral'
                              }

        self.test_db_path = os.path.join(os.path.dirname(__file__), 
                                         'datasets/test_db.sqlite')

#         self.dataset.split_dataset()
#         self.train_dataset = self.dataset.get_datasplit('train')
#         self.validate_dataset = self.dataset.get_datasplit('validate')
#         self.test_dataset = self.dataset.get_datasplit('test')
        
    #------------------------------------
    # testDistributedSampling
    #-------------------

    @unittest.skipIf(not TEST_ALL, 'Temporarily skip this test.')
    def testDistributedSampling(self):
        
        collected_samples = []
        if self.num_gpus == 0:
            print("No GPUs on this machine. Skipping distributed sampling test")
        for local_rank in range(self.num_gpus):
            os.environ['LOCAL_RANK'] = f"{local_rank}"
            samples = self.simulate_one_GPU_process()
            collected_samples.append(samples)

        self.output_check(collected_samples)

    #------------------------------------
    # simulate_one_GPU_process
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, 'Temporarily skip this test.')
    def simulate_one_GPU_process(self):

        world_size = int(os.environ['WORLD_SIZE'])
        node_rank  = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        cuda.set_device(local_rank)
        
        dataset = SqliteDataset(
            'fake/csv/path',
            self.label_mapping,
            sqlite_path=self.test_db_path,
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
        accumulated_data = {'epoch0' : [],
                            'epoch1' : [],
                            }
        for epoch in range(2):
            self.dataloader.set_epoch(epoch)
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

    #------------------------------------
    # output_check
    #-------------------

    def output_check(self, accumulated_samples):
        
        epoch0_samples = []
        epoch1_samples = []
        
        for process_collected_samples in accumulated_samples:
            # Dict with epoch0 and epoch1 samples
            # collected by one process:
            epoch0_samples.extend(process_collected_samples['epoch0'])
            epoch1_samples.extend(process_collected_samples['epoch1'])
        
        num_samples = len(self.dataloader)
        self.assertEqual(len(epoch0_samples), num_samples)
        self.assertEqual(len(epoch1_samples), num_samples)
        self.assertEqual(epoch0_samples, epoch1_samples)
        self.assertEqual(sorted(epoch0_samples) == range(num_samples))
        self.assertEqual(sorted(epoch1_samples) == range(num_samples))

    #------------------------------------
    # run_through_samples 
    #-------------------

#     def run_through_samples(self,node_rank, df):
#     
#         for epoch in range(3):
#             self.dataloader.set_epoch(epoch)
#             df.loc[f"node{node_rank}"][f"epoch{epoch}"] = []
#             for res_dict in self.dataloader:
#                 val = res_dict['tok_ids'][0][0]
#                 df.loc[f"node{node_rank}"][f"epoch{epoch}"].append(int(val))
#         return df


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

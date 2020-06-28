#!/usr/bin/env python3
'''
Created on Jun 23, 2020

@author: paepcke
'''
import os
import sqlite3
import subprocess
import tempfile
import unittest

import GPUtil

from bert_feeder_dataloader import MultiprocessingDataloader
from bert_feeder_dataset import SqliteDataset


#*******
# The following is a workaround for some library
# setting the env var to 'Intel' and causing
# an error.
os.environ['MKL_THREADING_LAYER'] = 'gnu'
#*******


#******TEST_ALL = True
TEST_ALL = False

class MultiProcessSamplerTester(unittest.TestCase):
    
    test_db_path = os.path.join(os.path.dirname(__file__), 'datasets/test_db.sqlite')
    launch_script_path = os.path.join(os.path.dirname(__file__), 'launch.py')
    runtime_script     = os.path.join(os.path.dirname(__file__), 'training_script_test_helper.py')

    #------------------------------------
    # setupClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        super(MultiProcessSamplerTester, cls).setUpClass()
        
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

#*************
#         if cls.num_gpus > 0:
#             torch.distributed.init_process_group(backend="nccl")
#*************

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

    #*****@unittest.skipIf(not TEST_ALL, 'Temporarily skip this test.')
    def testDistributedSampling(self):

        # Temp file template from which to derive
        # the result files where each forked process
        # puts its samples dict:

        if self.num_gpus == 0:
            print("No GPUs on this machine. Skipping distributed sampling test")

        try:
            tmpdirname = tempfile.TemporaryDirectory(prefix='Results',dir='/tmp')
            for local_rank in range(self.num_gpus):
                os.environ['LOCAL_RANK'] = f"{local_rank}"
    
                # Launch processes:
                completed_process = subprocess.run([self.launch_script_path,
                                                    self.runtime_script,
                                                    tmpdirname
                                                    ])
                if completed_process.returncode != 0:
                    print("*********Non zero return code from launch")
    
    
            self.output_check(tmpdirname)
        finally:
            os.removedirs(tmpdirname)

 
    #------------------------------------
    # output_check
    #-------------------

    def output_check(self, tmpdirname):
        
        epoch0_samples = []
        epoch1_samples = []
        
        for local_rank in range(self.num_gpus):
            res_file = os.path.join(tmpdirname, f"_{local_rank}.txt")
            with open(res_file, 'r') as fd:
                res_dict_str = fd.read()
                res_dict = eval(res_dict_str,
                                {"__builtins__":None},    # No built-ins at all
                                {}                        # No additional func
                                )
                epoch0_samples.extend(res_dict['epoch0'])
                epoch1_samples.extend(res_dict['epoch1'])
        
        num_samples = len(self.dataloader)
        self.assertEqual(len(epoch0_samples), num_samples)
        self.assertEqual(len(epoch1_samples), num_samples)
        self.assertEqual(epoch0_samples, epoch1_samples)
        self.assertEqual(sorted(epoch0_samples) == range(num_samples))
        self.assertEqual(sorted(epoch1_samples) == range(num_samples))



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

#!/usr/bin/env python3
'''
Created on Jun 23, 2020

@author: paepcke
'''
import numpy as np

import os
import subprocess
import sqlite3
import unittest
import pandas as pd
import GPUtil

#*******
# The following is a workaround for some library
# setting the env var to 'Intel' and causing
# an error.
os.environ['MKL_THREADING_LAYER'] = 'gnu'
#*******
from bert_feeder_dataloader import MultiprocessingDataloader
from bert_feeder_dataset import SqliteDataset

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
        cls.db.execute('''CREATE TABLE Samples (tok_ids varchar(50),
                                                attention_mask varchar(50),
                                                label int
                                                )

                                               ''')
        # 20 samples:
        for serial_num in range(20):
            cls.db.execute(f'''INSERT INTO Samples
                            VALUES("[{serial_num+1}]",
                                   "[0,1,0,0]",
                                  {serial_num+1}
                                  )
                            '''
            )

        cls.db.commit()

    #------------------------------------
    # setUp 
    #-------------------
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        label_mapping = {0, 'right',
                         1, 'left',
                         2, 'neutral'
            }
        self.dataset = SqliteDataset(
            'fake/csv/path',
            label_mapping,
            sqlite_path=self.test_db_path,
            sequence_len=128,
            text_col_name='foo',
            label_col_name='bar',
            delete_db=False,
            quiet=True
            )        
        
        self.dataset.split_dataset()
        self.train_dataset = self.dataset.get_datasplit('train')
        self.validate_dataset = self.dataset.get_datasplit('validate')
        self.test_dataset = self.dataset.get_datasplit('test')
        
    #------------------------------------
    # testDistributedSampling
    #-------------------

    def testDistributedSampling(self):
        if self.num_cuda_devs == 0:
            print("Skipping testDistributedSampling because no GPUs on this machine")
            return True

        # The various processes will write the data collected
        # during epoch0 to /tmp/epoch0.output, and the data
        # collected during epoch1 to /tmp/epoch1.output. Delete
        # those files first:

        try:
          os.remove('/tmp/epoch0.output')
        except FileNotFoundError:
          pass
        try:
            os.remove('/tmp/epoch1.output')
        except FileNotFoundError:
          pass

        # Run launch.py, passing it the path to
        # 'test_multiprocess_sampler_helper.py' as the
        # script each process is to run. In a real system
        # That would be the training script. This one
        # has each process ask the dataloader for randomized
        # samples that it can run on its GPUs, writing
        # results to /tmp/epoch{0|1}.output.

        # The launch.py script by default uses all GPUs
        # on its machine:
        
        completed_proc = subprocess.run(['python3', self.launch_script_path, self.runtime_script])

        # Ensure:
        #   1. that among all processes both epochs cover
        #      the entire dataset
        #   2. that the dataset is traversed in a different
        #      order between epoch 0 and epoch1
        #   3. all GPUs are used
        
        self.assertTrue(self.output_check(0))
        self.assertTrue(self.output_check(1))
    
    #------------------------------------
    # output_check
    #-------------------

    def output_check(self, epoch):
        '''
        Check /tmp/epoch0.output and /tmp/epoch1.output
        to see wheter:

           1. that among all processes both epochs cover
              the entire dataset
           2. that the dataset is traversed in a different
              order between epoch 0 and epoch1
           3. all GPUs are used

        The output files look like this:

        Node2 GPU2 14
        Node2 GPU2 15
           ...
        Node1 GPU1 6
        Node1 GPU1 8
           ...
        Node0 GPU0 20
        Node0 GPU0 4
           ...

        '''

        nodes_used   = set()
        gpus_used    = set()
        data_sampled = []

        outfile = f"/tmp/epoch{epoch}.output"
        with open(outfile, 'r') as epoch_fd:
            for line in epoch_fd:
                (node, gpu, sample) = line.split(' ')
                node_num = node[-1]
                gpu_num  = gpu[-1]
                nodes_used.add(node_num)
                gpus_used.add(gpu_num)
                data_sampled.append(int(sample))

            # Did every node 0-2 contribute to epoch0?
            self.assertEqual(len(nodes_used), 3, f"Only nodes {nodes_used} were used; should be 3.")
            self.assertEqual(len(gpus_used),
                             self.num_cuda_devs,
                             f"Only gpus {gpus_used} were used; should be {self.num_cuda_devs}."
                             )

            sorted_data = sorted(data_sampled)
            expected    = list(np.array(range(20)) + 1)
            self.assertEqual(expected, sorted_data)
        return True

    #------------------------------------
    # run_through_samples 
    #-------------------

    def run_through_samples(self,node_rank, df):
    
        for epoch in range(3):
            self.dataloader.set_epoch(epoch)
            df.loc[f"node{node_rank}"][f"epoch{epoch}"] = []
            for res_dict in self.dataloader:
                val = res_dict['tok_ids'][0][0]
                df.loc[f"node{node_rank}"][f"epoch{epoch}"].append(int(val))
        return df


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

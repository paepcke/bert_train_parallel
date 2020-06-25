'''
Created on Jun 23, 2020

@author: paepcke
'''
import os
import sqlite3
import unittest
import pandas as pd
import numpy as np

from bert_feeder_dataloader import MultiprocessingDataloader
from bert_feeder_dataset import SqliteDataset


class MultiProcessSamperTester(unittest.TestCase):
    
    test_db_path = os.path.join(os.path.dirname(__file__), 'datasets/test_db.sqlite')

    #------------------------------------
    # setupClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        super(MultiProcessSamperTester, cls).setUpClass()
        # Create a test Sqlite db:
        try:
            os.remove(cls.test_db_path)
        except FileNotFoundError:
            pass
        cls.db = sqlite3.connect(cls.test_db_path)
        cls.db.row_factory = sqlite3.Row
        
#         cls.db.execute('''CREATE TABLE Samples (tok_ids varchar(50),
#                                                 attention_mask varchar(50),
#                                                 label int
#                                                 )
#                                                ''')
#         for serial_num in range(9):
#             cls.db.execute(f'''INSERT INTO Samples
#                             VALUES("[{serial_num},{serial_num+1},{serial_num+2}]",
#                                    "[0,1,0,0]",
#                                   {serial_num} % 3
#                                   )
#                             '''
#             )
        cls.db.execute('''DROP TABLE IF EXISTS Samples''')
        cls.db.execute('''CREATE TABLE Samples (tok_ids varchar(50),
                                                attention_mask varchar(50),
                                                label int
                                                )
                                               ''')
        for serial_num in range(19):
            cls.db.execute(f'''INSERT INTO Samples
                            VALUES("[{serial_num}]",
                                   "[0,1,0,0]",
                                  {serial_num}
                                  )
                            '''
            )
        
        
#        cls.db.execute('''CREATE TABLE Samples (content int)''')
#         for serial_num in range(9):
#             cls.db.execute(f'''INSERT INTO Samples
#                             VALUES({serial_num + 1})
#                             '''
#             )

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
    # testSampling
    #-------------------

    def testSampling(self):
        df = pd.DataFrame(index=['node0','node1', 'node2'],
                  columns=['epoch0', 'epoch1', 'epoch2'])

        node_rank = 0
        self.dataloader = MultiprocessingDataloader(
            self.train_dataset,
            #****3, # world_size
            2, 
            node_rank
            )
        df = self.run_through_samples(node_rank, df)

        # Pretend to be a different node (machine):
        node_rank = 1
        df = self.run_through_samples(node_rank, df)

        # Again: Pretend to be a different node (machine):
        node_rank = 2
        df = self.run_through_samples(node_rank, df)

        print(df)
        df_content = np.array([[8, 3, 4],[4, 5, 8],[0, 3, 2],
                               [0, 7, 5],[0, 7, 3],[8, 7, 5],
                               [2, 1, 6],[6, 2, 1],[6, 1, 4]
                               ]).reshape([3,3])        
        correct_df = pd.DataFrame(df_content,
                                  index=['node0','node1', 'node2'],
                                  columns=['epoch0', 'epoch1', 'epoch2'])
        self.assertEqual(df_content, correct_df)
 

    
    #------------------------------------
    # run_through_samples 
    #-------------------

    def run_through_samples(self,node_rank, df):
    
        for epoch in range(3):
            self.dataloader.set_epoch(epoch)
            df.loc[f"node{node_rank}"][f"epoch{epoch}"] = []
            for res_dict in self.dataloader:
#                 print(f"{res_dict['tok_ids']}|"
#                       f"{res_dict['attention_mask']}|" 
#                       f"{res_dict['label']}")
                val = res_dict['tok_ids'][0][0]
                
                df.loc[f"node{node_rank}"][f"epoch{epoch}"].append(int(val))
        return df


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
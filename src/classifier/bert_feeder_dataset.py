'''
Created on Jun 11, 2020

@author: paepcke
'''

import ast
from collections import deque
import csv
import os
import re
from sqlite3 import OperationalError as DatabaseError
import sqlite3
import subprocess
import sys

from pandas.core.frame import DataFrame
from torch.utils.data import Dataset

import numpy as np
from classifier.logging_service import LoggingService
from classifier.text_augmentation import TextAugmenter

TESTING = False

class BertFeederDataset(Dataset):
    '''
    Takes path to a CSV file prepared by ****????****
    Columns: id,advertiser,page,leaning,tokens,ids
    Sample:
      (10,'Biden','http://...','left',"['[CLS],'Joe','runs',...'[SEP]']",'[[114 321 ...],[4531 ...]])
    
    Tokens is a stringified array of array of tokens.
    Length: sequence size (e.g. 128)
    Length: as many are there are lines of sample ads.
    Ids are a stringified arrays of arrays of ints. Each
      int is an index into BERT vocab. 
    Length: sequence size (e.g. 128)
    
    Result in sqlite DB; acts as iterator, has len()
    '''

    SEQUENCE_LEN    = 128 
    TEXT_COL_NAME   = 'text'
    LABEL_COL_NAME  = 'label'
    IDS_COL_NAME    = 'tok_ids'
    
    SPACE_TO_COMMA_PAT = re.compile(r'([0-9])[\s]+')
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 csv_path,
                 sqlite_path=None,
                 sequence_len=None,
                 text_col_name=None,
                 label_col_name=None,
                 ):

        self.log = LoggingService()

        if text_col_name is None:
            self.text_col_name = self.TEXT_COL_NAME
        else:
            self.text_col_name = text_col_name
            
        if label_col_name is None:
            self.label_col_name = self.LABEL_COL_NAME
        else:
            self.text_col_name = text_col_name
            
        if sqlite_path is None:
            (file_path, _ext) = os.path.splitext(csv_path)
            sqlite_path = file_path + '.sqlite'

        #*********
        if os.path.exists(sqlite_path):
            os.remove(sqlite_path)
        #*********
            
        if os.path.exists(sqlite_path):
            self.log.info(f"Using existing db {sqlite_path} (not raw csv)")
            self.db = sqlite3.connect(sqlite_path)
            self.db.row_factory = sqlite3.Row 
            
        else:
            # Fill the sqlite db with records, each
            # containing sample_id, toc_ids, label, attention_mask.
            # Also sets self.samples_ids
            self.db = self.process_csv_file(csv_path,
                                            sqlite_path, 
                                            sequence_len,
                                            text_col_name,
                                            label_col_name
                                            )
        # Usually, caller will next call split_dataset(),
        # which will create queues of sample IDs for
        # train/val/test. Till then, the whole dataset is
        # the train queue:
        try:
            res = self.db.execute('''
                   SELECT ROWID AS sample_id 
                     FROM Samples''')
        except DatabaseError as e:
            self.log.err(f"Could not retrieve sample ids from db {sqlite_path}: {repr(e)}")
            sys.exit(1)

        self.train_queue = [row['sample_id'] for row in res.fetchall()]
        self.num_samples = len(self.train_queue)

    #------------------------------------
    # switch_to_split 
    #-------------------
    
    def switch_to_split(self, split_id):
        
        if split_id == 'train':
            self.curr_queue = self.train_queue
        elif split_id == 'validate':
            self.curr_queue = self.val_queue
        elif split_id == 'test':
            self.curr_queue = self.test_queue
        else:
            raise ValueError(f"Dataset ID must be one of train/validate/test; was {split_id}")

    #------------------------------------
    # curr_dataset_id 
    #-------------------

    def curr_split_id(self):
        if self.curr_queue == self.train_queue:
            return 'train'
        if self.curr_queue == self.val_queue:
            return 'validate'
        if self.curr_queue == self.test_queue:
            return 'test'
        raise ValueError("Bad curr_queue")
        
    #------------------------------------
    # seek 
    #-------------------

    def reset(self, split_id=None):
        '''
        Sets the dataset's queue to the beginning.
        If dataset_id is None, resets the current
        split.
                
        @param split_id:
        @type split_id:
        '''
        
        if split_id == 'train':
            self.train_queue = self.saved_queues['train']
        elif split_id == 'validate':
            self.val_queue = self.saved_queues['validate']
        elif split_id == 'test':
            self.test_queue = self.saved_queues['test']
        else:
            raise ValueError(f"Dataset ID must be one of train/validate/test; was {split_id}")


    #------------------------------------
    # process_csv_file 
    #-------------------
    
    def process_csv_file(self, 
                         csv_path,
                         sqlite_path,
                         sequence_len,
                         text_col_name,
                         label_col_name):
        '''
        Create an sqlite db containing table 
        'Samples' with cols
           sample_id  int
           tok_ids    str  e.g. '[254,456,...]'
           label      str
           attention_mask str   e.g. [1,0,0,1,...]
        
        CSV file must contain at least a column
        called self.text_col_name and self.table_col_name
        
        
        @param csv_path: 
        @type csv_path:
        @param sqlite_path:
        @type sqlite_path:
        @param sequence_len:
        @type sequence_len:
        @param text_col_name:
        @type text_col_name:
        @param label_col_name:
        @type label_col_name:
        '''
        
        # Set defaults where needed:
        if sequence_len is None:
            sequence_len = self.SEQUENCE_LEN
        if text_col_name is None:
            text_col_name = self.TEXT_COL_NAME
        if label_col_name is None:
            label_col_name = self.LABEL_COL_NAME

        self.sequence_len = sequence_len
        self.text_col_name = text_col_name
        self.label_col_name = label_col_name

        # Facility to tokenize and otherwise
        # convert samples to formats ready for
        # BERT:
        self.text_augmenter = TextAugmenter(sequence_len)
        # Get number of CSV lines:
        res = subprocess.run(['wc', '-l', csv_path], capture_output=True)
        # Returns something line: b'   23556 /Users/foo.csv':
        (num_csv_lines, _filename) = res.stdout.decode().strip().split(' ')
        csv_fd = open(csv_path, 'r')
        db = sqlite3.connect(sqlite_path)
        db.row_factory = sqlite3.Row
        # Note: SQLITE3 has automatic ROWID column, which 
        #       will serve as our key:
        db.execute('''DROP TABLE IF EXISTS Samples''')
        db.execute('''
                   CREATE TABLE Samples (
                      tok_ids text,
                      attention_mask text,
                      label int
                      )
                   ''')
        num_processed = 0
        try:
            self.reader = csv.DictReader(csv_fd)
            # Some texts are partitioned into 
            # multiple rows, if they exceed
            # sequence_len. A queue to manage
            # them:
            self.queued_samples = deque()
            while True:
                # Next dict with 'ids', 'label, 'attention_mask':
                # Will throw StopIteration when done:
                row_dict = self.next_csv_row()
                if row_dict is None:
                    # An error in the CSV file; next_csv_row()
                    # already wrote an error msg. Keep going
                    continue
                insert_cmd = f'''
                           INSERT INTO Samples (tok_ids, 
                                                attention_mask, 
                                                label
                                                ) 
                            VALUES (
                              '{str(row_dict['tok_ids'])}',
                              '{str(row_dict['attention_mask'])}',
                              {row_dict['label']}
                              )
                           '''
                db.execute(insert_cmd)
                num_processed += 1
                #************
                if TESTING:
                    if num_processed >= 100:
                        db.commit()
    
                        break
                #************
                if num_processed % 1000 == 0:
                    db.commit()
                    self.log.info(f"Processed {num_processed}/{num_csv_lines} CSV records")
        finally:
            csv_fd.close()
            self.sample_ids = list(db.execute('''
                                              SELECT ROWID AS sample_id from Samples
                                              '''
                                              ))
        return db
    
    #------------------------------------
    # __next__ 
    #-------------------

    def __next__(self):
        next_sample_id = self.curr_queue.popleft()
        res = self.db.execute(f'''
                               SELECT tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE ROWID = {next_sample_id}
                             ''')
        row = next(res)
        return self.clean_row_res(dict(row))
    
    #------------------------------------
    # __getitem__ 
    #-------------------

    def __getitem__(self, indx):
        '''
        Return indx'th row from the db.
        The entire queue is always used,
        rather than the remaining queue
        after some popleft() ops. 
        
        @param indx:
        @type indx:
        '''

        ith_sample_id = self.saved_queues[self.curr_split_id()][indx]
        res = self.db.execute(f'''
                               SELECT tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE ROWID = {ith_sample_id}
                             ''')
        # Return the (only result) row:
        row = next(res)
        return self.clean_row_res(dict(row))
    
    #------------------------------------
    # __iter__ 
    #-------------------
    
    def __iter__(self):
        return self

    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        '''
        Return length of the current split. Use
        switch_to_split() before calling this
        method to get another split's length.
        The length of the entire queue is returned,
        not just what remains after calls to next()
        '''
        return len(self.saved_queues[self.curr_split_id()])

    #------------------------------------
    # next_csv_row
    #-------------------
 
    def next_csv_row(self):
        '''
        Returns a dict 'ids', 'label', 'attention_mask'
        '''

        # Still have a row left from a previouse
        # chopping?
        if len(self.queued_samples) > 0:
            return(self.queued_samples.popleft())
        
        # No pending samples from previously
        # found texts longer than sequence_len:
        row = next(self.reader)
        txt = row[self.text_col_name]

        # Tokenize the text of the row (the ad):
        # If the ad is longer than self.SEQUENCE_LEN,
        # then multiple rows are returned.
        # Each returned 'row' is a dict containing
        # just the key self.IDS_COL. Its value is
        # an array of ints: each being an index into
        # the BERT vocab.
        #
        # The ids will already be padded. Get
        #   [{'ids' : [1,2,...]},
        #    {'ids' : [30,64,...]}
        #        ...
        #   ]
        
        # Get list of dicts: {'tokens' : ['[CLS]','foo',...'[SEP]'],
        #                     'ids'    : [2545, 352, ]
        #                    }
        # dicts. Only one if text is <= sequence_len, else 
        # more than one:
        id_dicts = self.text_augmenter.fit_one_row_to_seq_len(txt) 

        # Add label. Same label even if given text was
        # chopped into multiple rows b/c the text exceeded
        # sequence_len:
        
        label = row[self.label_col_name]
        if label == 'right':
            label_encoding = 0
        elif label == 'left':
            label_encoding = 1
        elif label == 'neutral':
            label_encoding = 2
        else:
            self.log.err(f"Unknown label encoding: {label}")
            return
            
        
        for id_dict in id_dicts:
            id_dict['label'] = label_encoding 

        # Create a mask of 1s for each token followed by 0s for padding
        for ids_dict in id_dicts:
            ids_seq = id_dict[self.IDS_COL_NAME]
            #seq_mask = [float(i>0) for i in seq]
            seq_mask = [int(i>0) for i in ids_seq]
            ids_dict['attention_mask'] = seq_mask

        # We now have a list of dicts, each with three
        # keys: 'ids','label','attention_mask'
        if len(id_dicts) > 1:
            self.queued_samples.extend(id_dicts[1:])
        return id_dicts[0]

    #------------------------------------
    # split_dataset 
    #-------------------
    
    def split_dataset(self,
                      sample_ids_or_df=None, 
                      train_percent=0.8,
                      val_percent=0.1,
                      test_percent=0.1,
                      random_seed=1845):

        if sample_ids_or_df is None:
            sample_ids_or_df = self.sample_ids
            
        # Deduce third portion, if one of the
        # splits is None:
        if train_percent is None:
            if val_percent is None or test_percent is None:
                raise ValueError("Two of train_percent/val_percent/test_percent must be non-None")
            train_percent = 1-val_percent-test_percent
        elif val_percent is None:
            if train_percent is None or test_percent is None:
                raise ValueError("Two of train_percent/val_percent/test_percent must be non-None")
            val_percent = 1-train_percent-test_percent
        elif test_percent is None:
            if train_percent is None or val_percent is None:
                raise ValueError("Two of train_percent/val_percent/test_percent must be non-None")
            test_percent = 1-train_percent-val_percent
            
        if train_percent+val_percent+test_percent != 1.0:
            raise ValueError("Values for train_percent/val_percent/test_percent must add to 1.0")
            
        np.random.seed(random_seed)
        if type(sample_ids_or_df) == DataFrame:
            sample_indices = list(sample_ids_or_df.index) 
        else:
            sample_indices = sample_ids_or_df
             
        perm = np.random.permutation(sample_indices)
        # Permutations returns a list of arrays:
        #   [[12],[40],...]; turn into simple list of ints:
        perm = [sample_idx_arr[0] for sample_idx_arr in perm]
        num_samples = len(perm)
        
        train_end = int(train_percent * num_samples)
        validate_end = int(val_percent * num_samples) + train_end
        self.train_queue = deque(perm[:train_end])
        self.val_queue = deque(perm[train_end:validate_end])
        self.test_queue = deque(perm[validate_end:])
        
        self.curr_queue = self.train_queue
        
        self.saved_queues = {}
        self.saved_queues['train'] = self.train_queue.copy()
        self.saved_queues['validate'] = self.val_queue.copy()
        self.saved_queues['test'] = self.test_queue.copy()
        
        return (self.train_queue, self.val_queue, self.test_queue)

    #------------------------------------
    # save_queues 
    #-------------------
    
    def save_queues(self, train_queue, val_queue, test_queue):
        
        self.db.execute('DROP TABLE IF EXISTS TrainQueue')
        self.db.execute('CREATE TABLE TrainQueue (sample_id int)')

        self.db.execute('DROP TABLE IF EXISTS ValidateQueue')
        self.db.execute('CREATE TABLE ValidateQueue (sample_id int)')

        self.db.execute('DROP TABLE IF EXISTS TestQueue')
        self.db.execute('CREATE TABLE TestQueue (sample_id int)')
        
        # Turn [2,4,6,...] into tuples: [(2,),(4,),(6,),...]
        train_tuples = [(sample_id,) for sample_id in train_queue]
        self.db.executemany("INSERT INTO TrainQueue VALUES(?);", train_tuples)

        val_tuples = [(sample_id,) for sample_id in val_queue]
        self.db.executemany("INSERT INTO ValidateQueue VALUES(?);", val_tuples)

        test_tuples = [(sample_id,) for sample_id in test_queue]
        self.db.executemany("INSERT INTO TestQueue VALUES(?);", test_tuples)
        
        self.db.commit()
        
    #------------------------------------
    # to_np_array 
    #-------------------

    def to_np_array(self, array_string):
        '''
        Given a string:
          "[ 124  56  32]"
        return an np_array: np.array([124,56,32]).
        Also works for more reasonable strings like:
          "[1, 2, 5]"
        
        @param array_string: the string to convert
        @type array_string: str
        '''

        # Use the pattern to substitute occurrences of
        # "123   45" with "123,45". The \1 refers to the
        # digit that matched (i.e. the capture group):
        proper_array_str = self.SPACE_TO_COMMA_PAT.sub(r'\1,', array_string)
        # Remove extraneous spaces:
        proper_array_str = re.sub('\s', '', proper_array_str)
        # Turn from a string to array:
        return np.array(ast.literal_eval(proper_array_str))


    #------------------------------------
    # clean_row_res
    #-------------------
    
    def clean_row_res(self, row):
        '''
        Given a row object returned from sqlite, 
        turn tok_ids and attention_mask into real
        np arrays, rather than their original str
        
        @param row:
        @type row:
        '''
        
        # tok_ids are stored as strings:
        row['tok_ids'] = self.to_np_array(row['tok_ids'])
        row['attention_mask'] = self.to_np_array(row['attention_mask'])
        return row


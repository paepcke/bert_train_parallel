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
from logging_service import LoggingService
from text_augmentation import TextAugmenter

TESTING = False
#TESTING = True

# ------------------------------- Class ReadOnlyDataset ----------

class FrozenDataset(Dataset):

    SPACE_TO_COMMA_PAT = re.compile(r'([0-9])[\s]+')
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 log,
                 db,
                 split_id,
                 queue,
                 label_mapping,
                 sample_ids
                 ):
        self.log = log
        self.db = db
        self._split_id = split_id
        self.label_mapping = label_mapping
        self.sample_ids = sample_ids
                
        self.queue = queue
        self.saved_queue = queue.copy()

    #------------------------------------
    # split_id
    #-------------------
    
    def split_id(self):
        try:
            return self._split_id
        except AttributeError:
            return "Not Yet Split"

    #------------------------------------
    # reset
    #-------------------

    def reset(self):
        '''
        Sets the dataset's queue to the beginning.
        '''

        # Replenish the requested queue

        self.queue = self.saved_queue.copy()

# ---------------------- Utilities ---------------

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

    
    #------------------------------------
    # __next__ 
    #-------------------

    def __next__(self):
        try:
            next_sample_id = self.queue.popleft()
        except IndexError:
            raise StopIteration
        
        res = self.db.execute(f'''
                               SELECT sample_id, tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE sample_id = {next_sample_id}
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

        ith_sample_id = self.saved_queue[indx]
        res = self.db.execute(f'''
                               SELECT sample_id, tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE sample_id = {ith_sample_id}
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
        return len(self.saved_queue)

# ------------------------------- Class BertFeederDataset ----------

class SqliteDataset(FrozenDataset):
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

    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 csv_or_sqlite_path,
                 label_mapping,
                 sequence_len=None,
                 text_col_name=None,
                 label_col_name=None,
                 ):
        '''
        A dataset for the context of Bert training.        
        One usually interacts with an instance of this
        class through a BertFeederDataloader instance
        (see bert_feeder_dataloader.py).
        
        This class is a subclass of the torch.util.Dataset
        class, and behaves as such. It can act as a stream
        of input sentences, or be a dict-like data source.
        For the dict-like behavior: 
        
            my_dataset[row_num]
            
        For the stream behavior: treat my_dataset as an
        iterator. 
        
        An additional feature is the option for integrated
        train/validation/test splits. Calling split_dataset()
        internally produces input queues that feed three 
        iterators. Callers switch between these iterators via
        the switch_to_split() method. The splits can be reset
        to their beginnings using the reset() method.
        
        Takes a CSV file, and generates an Sqlite database
        that holds the integer indexes of the collection
        vocab into the BERT vocab, the tokens, and the
        labels. The CSV file can have arbitrary columns;
        only two are required: a column with the raw text
        to be processed through a BERT model, and a column
        with the true labels. The column names default to
        
          BertFeederDataset.TEXT_COL_NAME
          BertFeederDataset.LABEL_COL_NAME
          
        These defaults can be changed in the __init__() call
        or in the class variable init.
        
        The label_mapping must be an OrderedDict mapping
        the textual labels in the CSV file to integers 0,1,...
        
        Ex CSV:
        
          id,     message,       page,    leaning

         165,"We are the..." ,http://...,  left        
            ,"Foo is bar..." ,   ...    ,  right
                    ...
        
        In this example the important cols are 'message', and 'leaning
        the label_mapping might be:
        
            OrderedDict({'right'   : 0,
                         'left'    : 1,
                         'neutral' : 2})
        
        Sequence length is the maximum number of text input 
        tokens into the model in one input sentence. A 
        typical number is 128. If input texts in the CSV are 
        longer than sequence_len, one or more additional input 
        sentences are constructed with the same label as the
        long-text row. Shorter sequences are padded.
        
        @param csv_path: path to CSV file. If sqlite_path is
            provided, and exists, the database at that location
            is used, instead of importing the CSV file. If not,
            an Sqlite db will be created in the same dir as
            csv_path. 
        @type csv_path: str
        @param label_mapping: mapping from text labels to ints
        @type label_mapping: OrderedDict({str : int})
        @param sqlite_path: path where the Sqlite db will be created
        @type sqlite_path: str
        @param sequence_len: width of BERT model input sentences 
            in number of tokens.
        @type sequence_len: int
        @param text_col_name: CSV column that holds text to process
        @type text_col_name: str
        @param label_col_name: CSV column that holds labels.
        @type label_col_name: str
        @param quiet: don't ask for confirmation about existing sqlite file:
        @type quiet: bool
        @param delete_db: if True, delete Sqlite db that contains the csv
            content right from the start. If None, ask user on the command
            line
        @type delete_db: {None|bool}
        '''

        self.log = LoggingService()

        if text_col_name is None:
            self.text_col_name = self.TEXT_COL_NAME
        else:
            self.text_col_name = text_col_name
            
        if label_col_name is None:
            self.label_col_name = self.LABEL_COL_NAME
        else:
            self.text_col_name = text_col_name

        self.label_mapping = label_mapping

        if not os.path.exists(csv_or_sqlite_path):
            raise IOError(f"Data source {csv_or_sqlite_path} does not exist.")
        
        is_csv_source = csv_or_sqlite_path.endswith('.csv')
        
        if is_csv_source:
            # Remove any existing sqlite db that goes
            # with this CSV file:
            (file_path, _ext) = os.path.splitext(csv_or_sqlite_path)
            sqlite_path = file_path + '.sqlite'
            if os.path.exists(sqlite_path):
                os.remove(sqlite_path)
            # Fill the sqlite db with records, each
            # containing sample_id, toc_ids, label, attention_mask.
            self.db = self.process_csv_file(csv_or_sqlite_path,
                                            sqlite_path, 
                                            sequence_len,
                                            text_col_name,
                                            label_col_name
                                            )
                
        else:
            self.db = sqlite3.connect(csv_or_sqlite_path)
            self.db.row_factory = sqlite3.Row

        num_samples_row = next(self.db.execute('''SELECT COUNT(*) AS num_samples from Samples'''))
        num_samples = num_samples_row['num_samples']
        # Sqlite3 ROWIDs go from 1 to n
        self.sample_ids = list(range(num_samples))

        # Make a preliminary train queue with all the
        # sample ids. If split_dataset() is called later,
        # this queue will be replaced:
        self.train_queue = deque(self.sample_ids)
        self.curr_queue  = self.train_queue
        self.saved_queues = {}
        # Again: this saved_queues entry will be
        # replaced upon a split:
        self.saved_queues['train'] = self.train_queue.copy()
        self.num_samples = len(self.train_queue)

    #------------------------------------
    # train_set 
    #-------------------

    def get_datasplit(self, split_id):
        if split_id == 'train':
            return  self.train_frozen_dataset
        elif split_id == 'validate':
            return self.validate_frozen_dataset
        elif split_id == 'test':
            return self.test_frozen_dataset
        else:
            raise ValueError("Only train, validate, and test are valid split ids.")

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
    # reset
    #-------------------

    def reset(self, split_id=None):
        '''
        Sets the dataset's queue to the beginning.
        If dataset_id is None, resets the current
        split.
                
        @param split_id:
        @type split_id:
        '''

        # After replenishing the requested
        # queue, check whether that queue was
        # self.curr_queue. If so, change self.curr_queue
        # to point to the new, refilled queue. Else
        # self.curr_queue remains unchanged:
                
        if split_id == 'train':
            old_train = self.train_queue
            self.train_queue = self.saved_queues['train'].copy()
            if self.curr_queue == old_train:
                self.curr_queue = self.train_queue

        elif split_id == 'validate':
            old_val = self.val_queue
            self.val_queue = self.saved_queues['validate'].copy()
            if self.curr_queue == old_val:
                self.curr_queue = self.val_queue
            
        elif split_id == 'test':
            old_test = self.test_queue
            self.test_queue = self.saved_queues['test'].copy()
            if self.curr_queue == old_test:
                self.curr_queue = self.test_queue

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
        called self.text_col_name and self.table_col_name.
        
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
        @return: a database (connection) instance
        @rtype: sqlite3.Connection
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

        db.execute('''DROP TABLE IF EXISTS Samples''')
        db.execute('''
                   CREATE TABLE Samples (
                      sample_id int primary key,
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
                try:
                    row_dict = self.next_csv_row()
                except StopIteration:
                    break
                if row_dict is None:
                    # An error in the CSV file; next_csv_row()
                    # already wrote an error msg. Keep going
                    continue
                insert_cmd = f'''
                           INSERT INTO Samples (sample_id,
                                                tok_ids, 
                                                attention_mask, 
                                                label
                                                ) 
                            VALUES (
                              {num_processed},
                              '{str(row_dict['tok_ids'])}',
                              '{str(row_dict['attention_mask'])}',
                              {row_dict['label']}
                              )
                           '''
                db.execute(insert_cmd)
                num_processed += 1
                #************
                if TESTING:
                    if num_processed >= 10000:
                        db.commit()
    
                        break
                #************
                if num_processed % 1000 == 0:
                    db.commit()
                    self.log.info(f"Processed {num_processed}/{num_csv_lines} CSV records")
        finally:
            db.commit()
            csv_fd.close()

        return db
    
    #------------------------------------
    # __next__ 
    #-------------------

    def __next__(self):
        try:
            next_sample_id = self.curr_queue.popleft()
        except IndexError:
            raise StopIteration
        
        res = self.db.execute(f'''
                               SELECT sample_id, tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE sample_id = {next_sample_id}
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
                               SELECT sample_id, tok_ids,attention_mask,label
                                FROM Samples 
                               WHERE sample_id = {ith_sample_id}
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
        try:
            txt = row[self.text_col_name]
        except KeyError:
            msg = (f"\nCSV file does not have a column named '{self.text_col_name}'\n"
                    "You can invoke bert_train_parallel.py with --text\n"
                    "to specify col name for text, and --label to speciy\n"
                    "name of label column."
                    )
            self.log.err(msg)
            raise ValueError(msg)

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

        try:        
            label = row[self.label_col_name]
        except KeyError:
            msg = f"CSV file does not have col {self.label_col_name}" + '\n' +\
                    "You can invoke bert_train_parallel.py with --label"
            self.log.err(msg)
            raise ValueError(msg)

        try:
            label_encoding = self.label_mapping[label]
        except KeyError:
            # A label in the CSV file that was not
            # anticipated in the caller's label_mapping dict
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
                      save_to_db=True,
                      random_seed=1845):
        '''
        Splits dataset into train, validation, and 
        test sets at the given proportions. One of the
        proportions may be set to None. In that case
        only two splits will be created. Randomly permutes
        samples before splitting
        
        The sample_ids_or_df may be a list of of
        indices into the sqlite db of sample rows from
        the original CSV file, or a dataframe in which 
        each row corresponds to a sample row from the 
        original CSV. If None, uses what this instance
        already knows. If in doubt, let it default.
        
        Creates a deque (a queue) for each split, and
        saves copies of each in a dict (saved_queues).
        Returns a triplet with the queues. 
        
        @param sample_ids_or_df: list of sqlite sample_id, or dataframe
        @type sample_ids_or_df: {list|pandas.dataframe}
        @param train_percent: percentage of samples for training
        @type train_percent: float
        @param val_percent: percentage of samples for validation
        @type val_percent: float
        @param test_percent: percentage of samples for testing
        @type test_percent: float
        @param save_to_db: whether or not to save the indices that
            define each split in the Sqlite db
        @type save_to_db: bool
        @param random_seed: seed for permuting dataset before split
        @type random_seed: int
        '''

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
        num_samples = len(perm)
        
        train_end = int(train_percent * num_samples)
        validate_end = int(val_percent * num_samples) + train_end
        self.train_queue = deque(perm[:train_end])
        self.val_queue = deque(perm[train_end:validate_end])
        self.test_queue = deque(perm[validate_end:])
        
        self.curr_queue = self.train_queue
        
        if save_to_db:
            self.save_queues(self.train_queue, self.val_queue, self.test_queue) 
        
        self.saved_queues = {}
        self.saved_queues['train'] = self.train_queue.copy()
        self.saved_queues['validate'] = self.val_queue.copy()
        self.saved_queues['test'] = self.test_queue.copy()
        
        self.train_frozen_dataset = FrozenDataset(self.log,
                                                  self.db,
                                                  'train',
                                                  self.saved_queues['train'],
                                                  self.label_mapping,
                                                  self.sample_ids
                                                  )
        
        self.validate_frozen_dataset = FrozenDataset(self.log,
                                                     self.db,
                                                     'validate',
                                                     self.saved_queues['validate'],
                                                     self.label_mapping,
                                                     self.sample_ids
                                                     )
        
        self.test_frozen_dataset = FrozenDataset(self.log,
                                                 self.db,
                                                 'test',
                                                 self.saved_queues['test'],
                                                 self.label_mapping,
                                                 self.sample_ids
                                                 )
        
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
        train_tuples = [(int(sample_id),) for sample_id in train_queue]
        self.db.executemany("INSERT INTO TrainQueue VALUES(?);", train_tuples)

        val_tuples = [(int(sample_id),) for sample_id in val_queue]
        self.db.executemany("INSERT INTO ValidateQueue VALUES(?);", val_tuples)

        test_tuples = [(int(sample_id),) for sample_id in test_queue]
        self.db.executemany("INSERT INTO TestQueue VALUES(?);", test_tuples)
        
        self.db.commit()

    #------------------------------------
    # save_dict_to_table 
    #-------------------
    
    def save_dict_to_table(self, table_name, the_dict, delete_existing=False):
        '''
        Given a dict, save it to a table in the underlying
        database.
        
        If the table exists, action depends on delete_existing.
        If True, the table is deleted first. Else the dict values
        are added as rows. 
        
        It is the caller's responsibility to ensure that:
        
           - Dict values are db-appropriate data types: int, float, etc.
           - The table name is a legal Sqlite table name  
        
        @param table_name: name of the table
        @type table_name: str
        @param dict: col/value information to store
        @type dict: {str : <any-db-appropriate>}
        '''
        if delete_existing:
            self.db.execute(f'''DROP TABLE IF EXISTS {table_name}''')
            self.db.execute(f'''CREATE TABLE {table_name} ('key_col' varchar(255),
                                                          'val_col' varchar(255));''')
            self.db.commit()

        insert_vals = list(the_dict.items())
        self.db.executemany(f"INSERT INTO {table_name} VALUES(?,?);", insert_vals)
        self.db.commit()

    #------------------------------------
    # yes_no_question 
    #-------------------

    def query_yes_no(self, question, default='yes'):
        '''
        Ask a yes/no question via raw_input() and return their answer.
    
        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
    
        The "answer" return value is True for "yes" or False for "no".
        '''
        valid = {"yes": True, "y": True, "ye": True,
                 "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)
    
        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                sys.stdout.write("Please respond with 'yes' or 'no' "
                                 "(or 'y' or 'n').\n")        

'''
Created on Jun 12, 2020

@author: paepcke
'''
from contextlib import contextmanager

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SqliteDataLoader(DataLoader):
    '''
    A dataloader that works with instances of 
    BertFeederDataset (see bert_feeder_dataset.py).
    This class simply wraps such a dataset. See
    header comment in that file for lots more
    information, such as dataset splitting, 
    switching between the splits, interacting
    either as a stream or a dict. 
    
    Instances can be used like any other Pytorch
    dataloader.
    
    This class adds a dataset split context manager.
    It allows callers to interact temporarily with 
    a particular split: test/validate/train, and
    then return to the current split. Example:
    
      with set_split_id('validate'):
          avg_val_accuracy = total_eval_accuracy / len(dataloader)
    
    '''
    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.my_split = dataset.split_id()

    #------------------------------------
    # split 
    #-------------------

    # No longer allowed. Instance is now initialized
    # with a frozen sub-collection.
    #def split_dataset(self, *args, **kwargs):
    #    self.dataset.split_dataset(*args, sample_ids_or_df=None, **kwargs)

    #------------------------------------
    # curr_split 
    #-------------------
    
    def split_id(self):
        '''
        Return this loader's dataset split id: 'train',
        'validate', or 'test'
        '''
        return self.my_split
        #return self.dataset.curr_split_id()

    #------------------------------------
    # reset_split
    #-------------------

    def reset_split(self):
        '''
        Sets the dataset's queue to the
        start.
        '''
        self.dataset.reset()

    #------------------------------------
    # __len__ 
    #-------------------

    def __len__(self):
        return len(self.dataset)
    
    #------------------------------------
    # __getitem__
    #-------------------

    def __getitem__(self, indx):
        return self.dataset[indx]

# -------------------- Multiprocessing Dataloader -----------

class MultiprocessingDataloader(SqliteDataLoader):
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, dataset, world_size, node_rank, **kwargs):
        
        self.dataset  = dataset
        
        self.sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=node_rank
                )

        super().__init__(dataset,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True,
                         sampler=self.sampler,
                         **kwargs)

    #------------------------------------
    # set_epoch 
    #-------------------

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)

# ------------------------ set_split_id Context Manager



#------------------------------------
# set_split_id [Context manager] 
#-------------------
    
@contextmanager
def set_split_id(dataloader, tmp_split_id):
    '''
    Allows temporary setting of split_id like this:
    
      with set_split_id(dataloader, 'validate'):
          dataloader.reset_split()
          
    or: get the validate split's length:

      with set_split_id('validate'):
          avg_val_accuracy = total_eval_accuracy / len(dataloader)
          
    The above temporarily sets the dataloader's split
    to 'validate' for the duration of the 'with' body.
    Then the split is returned to the original value.
    
    @param dataloader: dataloader whose split is to be 
        temporarily changed. 
    @type dataloader: BertFeederDataloader
    @param tmp_split_id: the split id to which the dataloader
        is to be set for the scope of the with statement
    @type tmp_split_id: {'train'|'validate'|'test'}
    '''
    saved_split_id = dataloader.curr_split()
    dataloader.switch_to_split(tmp_split_id)
    try:
        yield
    finally:
        dataloader.switch_to_split(saved_split_id)


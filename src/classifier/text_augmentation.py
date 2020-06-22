'''
Created on Jun 7, 2020

@author: paepcke
'''

import os
import re
import sys

from pytorch_pretrained_bert import BertTokenizer

import pandas as pd

class TextAugmenter(object):
    '''
    Minimal columns: 'text', 'label'
    '''

    # Min number of tokens in a sequence to
    # consider the row for augmentation:
    MIN_AUG_LEN = 80 
    
    DEFAULT_SEQUENCE_LEN = 128
    
    # regex pattern to find newlines:
    NL_PAT = re.compile(r'\n')
    


    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 train_files, 
                 sequence_len=None,
                 outfile=None,
                 text_col='text',
                 label_col='label',
                 remove_txt_col=True,
                 in_memory=False,
                 testing=False):
        '''
        Constructor
        '''
        if outfile is None:
            outfile = os.path.join(os.path.dirname(__file__), 'tokenized_input.csv')
        if sequence_len is None:
            sequence_len = TextAugmenter.DEFAULT_SEQUENCE_LEN
        self.text_col  = text_col
        self.label_col = label_col
        self.tokens_col_name = 'tokens'
        self.ids_col_name   = 'tok_ids'
        
        self.sequence_len = sequence_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        if not in_memory:
            # If this instance will be used with a dataloader,
            # we are done  
            return

        # Do entire training set preparation in memory:
        self.train_df = self.read_files(train_files)
        if testing:
            # Let unittests call the other methods.
            return
        chopped_tokenized = self.fit_to_sequence_len(self.train_df)
        if remove_txt_col:
            chopped_tokenized = chopped_tokenized.drop(self.text_col, axis=1)
            
        # Add BERT ids column:
        ids = self.padded_ids(chopped_tokenized[self.tokens_col_name].values)
        chopped_tokenized['ids'] = ids
        chopped_tokenized.to_csv(outfile,index=False)
        
    #------------------------------------
    # fit_to_sequence_len 
    #-------------------
    
    def fit_to_sequence_len(self, train_df):
        '''
        Modify a df holding an entire training set:
           - Add a new column: 'tokens'
           - Have each row's number of tokens be
               at most self.seq_len. 
        
        @param train_df:
        @type train_df:
        '''
        # Add a col to the passed-in df: 'tokens'.
        # Create list the height of train_df:
        token_col = ['']*len(train_df)
        train_df[self.tokens_col_name] = token_col
        
        new_rows = []
        nl_pat = re.compile(r'\n')
        for (_indx, row) in train_df.iterrows():
            # Remove \n chars;
            try:
                txt = nl_pat.sub(' ', row[self.text_col])
            except TypeError:
                # Some csv entries have empty text cols.
                # Those manifest as NaN vals in the df.
                # Just skip those rows:
                continue
            # And chop into a seq of strings:
            tokenized_txt = self.tokenizer.tokenize(txt)
            
            # Short enough to just keep?
            # The -2 allows for the [CLS] and [SEP] tokens:
            if len(tokenized_txt) <= self.sequence_len - 2:
                row[self.tokens_col_name] = ['[CLS]'] + tokenized_txt + ['[SEP]']
                new_rows.append(row)
                continue

            # Go through the too-long tokenized txt, and cut into pieces
            # in which [CLS]<tokens>[SEP] are sequence_len long:
            for pos in range(0,len(tokenized_txt),self.sequence_len-2):
                sent_fragment = ['[CLS]'] + \
                                tokenized_txt[pos:pos+self.sequence_len-2]  + \
                                ['[SEP]']
                # Make a copy of the row, and fill in the token:
                new_row = row.copy()
                new_row[self.tokens_col_name] = sent_fragment
                # Add to the train_df:
                new_rows.append(new_row)
        new_rows_df = pd.DataFrame(new_rows, columns=train_df.columns)
        return new_rows_df

    #------------------------------------
    # fit_one_row_to_seq_len
    #-------------------
    
    def fit_one_row_to_seq_len(self, text):
        '''
        Takes a dict created by CSV reader. Dict may
        have many keys, but only one named txt_col_name
        is required. Default ids_col_name: self.ids_col_names
        
        Returns an array of dicts. Each dict contains
        key ids_col_name. The value is an array of ints:
        indices into BERT vocab. Each int corresponds to one
        token in the given text.
        
        @param text: the text to tokenize and create BERT ids for
        @type str
        @return: list of dicts: {'ids' : [2534,15,...]'}
        @rtype: ({str : arr[int]}
        '''

        # A single row may be chopped into multiple rows:
        new_rows = []
        # Remove \n chars;
        try:
            txt = self.NL_PAT.sub(' ', text)
        except TypeError:
            return None
        # And chop into a seq of strings:
        tokenized_txt = self.tokenizer.tokenize(txt)
        
        # Short enough to just keep?
        # The -2 allows for the [CLS] and [SEP] tokens:
        if len(tokenized_txt) <= self.sequence_len - 2:
            tokens = ['[CLS]'] + tokenized_txt + ['[SEP]']
            ids    = self.padded_ids(tokens)
            new_rows.append({self.ids_col_name : ids})
        else:
            # Go through the too-long tokenized txt, and cut into pieces
            # in which [CLS]<tokens>[SEP] are sequence_len long:
            for pos in range(0,len(tokenized_txt),self.sequence_len-2):
                sent_fragment = ['[CLS]'] + \
                                tokenized_txt[pos:pos+self.sequence_len-2]  + \
                                ['[SEP]']
                ids    = self.padded_ids(sent_fragment)
                new_rows.append({self.ids_col_name : ids})
        
        return new_rows

    #------------------------------------
    # augment_text 
    #-------------------

    def augment_text(self, train_df):
        '''
        Given a df with at least a column
        called 'tokens': find rows with more
        than MIN_AUG_LEN tokens. Select sequences
        that contain whole sentences, i.e. punctuation
        {.|,|!|?}. Then create new rows with all
        cols the same, except for the tokens column.
        If there is a self.text_col, its content
        will be the assembled clear text from the 
        tokens. Though token oddities will be present. 
           
        @param train_df:
        @type train_df:
        @return: new df with additional rows.
        @rtype: DataFrame
        '''
        
        #end_sentence_punctuation = '.!?'
        for (_indx, row) in train_df.iterrows():
            # Long enough token seq?
            if len(row[self.tokens_col_name]) < TextAugmenter.MIN_AUG_LEN:
                continue
            # sentence_bounds = self.get_indexes(row, end_sentence_punctuation)
            
            

    #------------------------------------
    # padded_id
    #-------------------
    
    def padded_ids(self, token_seq):
        '''
        Takes a list of tokens, and returns a list
        of integer ids. The ids are indices into 
        Bert vocabulary.
        
        @param token_seq:
        @type token_seq:
        @return: array of Bert ids
        @rtype: [int]
        '''
#         if type(token_seqs) != list:
#             token_seqs = [token_seqs]
#         ids = [self.tokenizer.convert_tokens_to_ids(tok_seq) \
#                for tok_seq in token_seqs]
        ids = self.tokenizer.convert_tokens_to_ids(token_seq)
        padded_ids = self.pad_sequences([ids],
                                        self.sequence_len,
                                        0
                                        ) 

        # Padded_ids is a 1-tuple. Inside is 
        # a numpy array.
        return padded_ids[0]

    #------------------------------------
    # pad_sequences 
    #-------------------
    
    def pad_sequences(self,
                      arrays,
                      targetlen,
                      fill_constant
                      ):
        new_arr = []
        for arr in arrays:
            arr_len = len(arr)
            if arr_len == targetlen:
                new_arr.append(arr)
                continue
            if arr_len > targetlen:
                new_arr.append(arr[:targetlen])
                continue
            # Sequence too short:
            filler = [fill_constant]*(targetlen - arr_len)
            arr.extend(filler)
            new_arr.append(arr)
        return new_arr

    #------------------------------------
    # get_indexes 
    #-------------------

    def get_indexes(self, arr, search_str):
        '''
        Function that returns the indexes of occurrences
        of any members in a given list within another list:
        
          Given arr = ['Earth', 'Moon', 'Earth']
          
        get_indexes(arr, 'Earth')   ==> [0,2]
        get_indexes(arr, ['Earth']) ==> [0,2]
        get_indexes(arr, ['Earth', 'Moon']) ==> [0,1,2]
        
                arr = ['[CLS]', 'The', 'Sun', '!', '[SEP]']
        
        get_indexes(arr, '.!?') ==> [3]
        
        @param arr: list within which to search
        @type arr: (<any>)
        @param search_str: what to search for
        @type search_str: str
        '''

        return [i for i in range(len(arr)) if arr[i] == search_str]


    #------------------------------------
    # read_files 
    #-------------------
    
    def read_files(self, train_files):
        df = pd.DataFrame()
        if not type(train_files) == list:
            train_files = [train_files]
        for train_file in train_files:
            df_tmp = pd.read_csv(train_file, 
                                 delimiter=',', 
                                 header=0,      # Col info in row 0
                                 quotechar='"',
                                 engine='python')
            df = pd.concat([df,df_tmp])

        return df

# ---------------------- Main ---------------

if __name__ == '__main__':
    in_csv_dir = "/Users/paepcke/EclipseWorkspacesNew/colab_server/src/jupyter/"
    
    # TRAINING:
#     train_files = [os.path.join(in_csv_dir, 'left_ads_final.csv'),
#                    os.path.join(in_csv_dir, 'right_ads_final.csv'),
#                    os.path.join(in_csv_dir, 'neutral_ads.csv'),
#                    os.path.join(in_csv_dir, 'combined-train.csv')
#                    ]
#     outfile = os.path.join(in_csv_dir, 'leanings_right_sized.csv')

    # TEST
    train_files   = os.path.join(in_csv_dir, 'final_test.csv')
    outfile = os.path.join(in_csv_dir, 'leanings_right_sized_testset.csv')
    
    if os.path.exists(outfile):
        print(f"Outfile {os.path.basename(outfile)} exists; please delete it and try again")
        sys.exit(1)
    
    TextAugmenter(train_files, outfile=outfile, text_col='message')
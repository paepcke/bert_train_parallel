#!/usr/bin/env python3
'''
Created on Jun 18, 2020

@author: paepcke
'''

import argparse
import os
import re
import sqlite3
import sys

import torch

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from logging_service import LoggingService


class BertResultAnalyzer(object):
    '''
    classdocs
    '''


    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, result_file):
        '''
        Constructor
        '''
        self.log = LoggingService()
        
        res_files_dict = self.get_result_file_paths(result_file)
               
        try:
            # Load the train/validate/test stats dict
            # from the db:
            stats_file = res_files_dict['stats_file']
            with open(stats_file, 'rb') as fd:
                # Load the data struct into cpu RAM, even
                # if it was on a GPU when it was saved:
                train_test_stats = torch.load(fd,
                                              map_location=torch.device('cpu')
                                              )
        except FileNotFoundError:
            self.log.err(f"No train/validate/test stats file found: {stats_file}")
            
        #****** Temporary Fix *******
        # Early version erroneously produced stats
        # dicts with key "Validation Accuracy" being 
        # set as "Validation Accuracy." (i.e. stray period)
        # Fix that here if needed:
        for epoch_res in train_test_stats['Training']:
            try:
                epoch_res['Validation Accuracy.']
                epoch_res['Validation Accuracy'] = epoch_res['Validation Accuracy.']
                del epoch_res['Validation Accuracy.']
            except KeyError:
                # All OK
                pass
        #****** End Temporary Fix **********
        
        # Print descriptives:
        db_file = res_files_dict['db_file']
        self.get_descriptives(train_test_stats,
                              db_file
                              )
 
        # Plot train/val losses by epoch:
        self.plot_train_val_loss_and_accuracy(train_test_stats)

    #------------------------------------
    # get_descriptives
    #-------------------
    
    def get_descriptives(self, 
                         train_test_stats, 
                         sqlite_file_path,
                         do_print=True):

        if not os.path.exists(sqlite_file_path):
            self.log.err(f"Sqlite file {sqlite_file_path} does not exist; no descriptives can be retrieved.")
            return
        test_res_dict = train_test_stats['Testing']
        try:
            db = sqlite3.connect(sqlite_file_path)
            res = db.execute('''SELECT label, count(*) AS num_this_label 
                                FROM Samples 
                               GROUP BY label;
                            ''')
            label_count_dict = {}
            # Build dict: string-label ===> number of samples
            for (int_label, num_this_label) in res:
                # Get str label from int label:
                str_label = next(db.execute(f'''SELECT "{int_label}" from LabelEncodings'''))
                label_count_dict[str_label] = num_this_label
        finally:
            db.close()
        
        if do_print:
            conf_mat = test_res_dict['Confusion matrix']
            del test_res_dict['Confusion matrix']
            train_res_df = pd.DataFrame(test_res_dict,
                                        index=[0])
            print(train_res_df)
            print(f"Confusion matrix:\n{conf_mat}")



    #------------------------------------
    # plot_train_val_loss_and_accuracy 
    #-------------------
    
    def plot_train_val_loss_and_accuracy(self, training_stats):
        '''
        View the summary of the training process.
        
        @param training_stats: a dict like:
           {
		     'Training' : [{'epoch': 1,
		                    'Training Loss': 0.016758832335472106,
		                    'Validation Loss': 0.102080237865448,
		                    'Training Accuracy': 0.00046875,
		                    'Validation Accuracy.': 0.05,
		                    'Training Time': '0:00:25',
		                    'Validation Time': '0:00:01'},
		                   {'epoch': 2,
		                      ...
		                   }
		                   ]
		   
		     'Testing'  : {'Test Loss': tensor(1.0733),
		                   'Test Accuracy': 0.1,
		                   'Matthews corrcoef': 0.0,
		                   'Confusion matrix': array([[0, 0, 0],
		                                              [3, 1, 6],
		                                              [0, 0, 0]])
		                  }
		   }

        @type training_stats_info: dict
        '''
        
        # Create a DataFrame from our training statistics.
        epoch_stats_dicts = training_stats['Training']
        
        #********
        # For testing when only one epoch's results
        # are available: add some more:
#         epoch_stats_dicts.extend(
#             [
#                 {'epoch': 2, 'Training Loss': 0.01250069046020508, 'Validation Loss': 0.0623801279067993, 'Training Accuracy': 0.01500625, 'Validation Accuracy.': 0.11, 'Training Time': '0:00:24', 'Validation Time': '0:00:01'},
#                 {'epoch': 3, 'Training Loss': 0.00250069046020508, 'Validation Loss': 0.0323801279067993, 'Training Accuracy': 0.02500625, 'Validation Accuracy.': 0.25, 'Training Time': '0:00:24', 'Validation Time': '0:00:01'},
#                 {'epoch': 4, 'Training Loss': 0.0004069046020508, 'Validation Loss': 0.0023801279067993, 'Training Accuracy': 0.01500625, 'Validation Accuracy.': 0.35, 'Training Time': '0:00:24', 'Validation Time': '0:00:01'}
#             ]
#             )
#         #********

        self.plot_stats_dataframe(epoch_stats_dicts)

    #------------------------------------
    # def plot_stats_dataframe 
    #-------------------

    def plot_stats_dataframe(self, epoch_stats_dicts):
        '''
        plot_type: 'loss' or 'accuracy'
        
        @param epoch_stats_dicts:
        @type epoch_stats_dicts:
        @param plot_type:
        @type plot_type:
        '''
        
        # Display floats with two decimal places.
        pd.set_option('precision', 2)

        df_stats = pd.DataFrame(epoch_stats_dicts)
        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')
        
        # A hack to force the column headers to wrap.
        #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
        
        # Display the table.
        df_stats
        
        # If you notice that, while the training loss is 
        # going down with each epoch, the validation loss 
        # is increasing! This suggests that we are training 
        # our model too long, and it's over-fitting on the 
        # training data. 
        
        # Validation Loss is a more precise measure than accuracy, 
        # because with accuracy we don't care about the exact output value, 
        # but just which side of a threshold it falls on. 
        
        # If we are predicting the correct answer, but with less 
        # confidence, then validation loss will catch this, while 
        # accuracy will not.
        
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')
        
        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        
        _fig, (ax1, ax2) = plt.subplots(nrows=1, 
                                        ncols=2, 
                                        figsize=(12,6),
                                        tight_layout=True
                                        )
        
        # Plot the learning curve.

        ax1.plot(df_stats['Training Loss'], 'b-o', label="Training")
        ax1.plot(df_stats['Validation Loss'], 'g-o', label="Validation")
        # Label the plot.
        ax1.set_title("Training & Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_xticks(df_stats.index.values)

        ax2.plot(df_stats['Training Accuracy'], 'b-o', label="Training")
        ax2.plot(df_stats['Validation Accuracy'], 'g-o', label="Validation")
        # Label the plot.
        ax2.set_title("Training & Validation Accuracy")
        ax1.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_xticks(df_stats.index.values)

        ax1.legend(frameon=False)
        ax2.legend(frameon=False)
        
        plt.show(block=False)

    #------------------------------------
    # get_result_file_paths 
    #-------------------
    
    def get_result_file_paths(self, one_result_file):
        '''
           facebook_ads_clean_testset_predictions.npy
           facebook_ads_clean_train_test_stats.dict
           facebook_ads_clean_trained_model.sav
           facebook_ads_clean.sqlite
           facebook_ads_clean.csv
           
        Returns:
                'preds_file' : file_path,
                'stats_file' : file_path,
                'model_file' : file_path,
                'db_file'    : file_path
                }


        @param one_result_file:
        @type one_result_file:
        '''
        preds_str = '_testset_predictions.npy'
        stats_str = '_train_test_stats.dict'
        model_str = '_trained_model.sav'
        db_str    = '.sqlite'
        
        if re.search(preds_str, one_result_file) is not None:
            files_root = one_result_file[:one_result_file.index(preds_str)]
        elif re.search(stats_str, one_result_file) is not None:
            files_root = one_result_file[:one_result_file.index(stats_str)]
        elif re.search(model_str, one_result_file) is not None:
            files_root = one_result_file[:one_result_file.index(model_str)]
        elif re.search(db_str, one_result_file) is not None:
            files_root = one_result_file[:one_result_file.index(db_str)]
        else:
            # Assume that caller gave the .csv file, or just the
            # path with the root:
            (files_root, _ext) = os.path.splitext(one_result_file) 
        
        return {'preds_file' : files_root + preds_str,
                'stats_file' : files_root + stats_str,
                'model_file' : files_root + model_str,
                'db_file'    : files_root + db_str
                }
        
    #------------------------------------
    # print_model_parms 
    #-------------------

    def print_model_parms(self, model):
        '''

        Printed out the names and dimensions of the weights for:
        
        1. The embedding layer.
        2. The first of the twelve transformers.
        3. The output layer.
        '''
        
        # Get all of the model's parameters as a list of tuples.
        params = list(model.named_parameters())
        
        self.log.info('The BERT model has {:} different named parameters.\n'.format(len(params)))
        self.log.info('==== Embedding Layer ====\n')
        for p in params[0:5]:
            self.log.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        self.log.info('\n==== First Transformer ====\n')
        for p in params[5:21]:
            self.log.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        self.log.info('\n==== Output Layer ====\n')
        for p in params[-4:]:
            self.log.info("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

# ------------------- Main ------------------

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Analyze results of Bert model runs."
                                     )

    parser.add_argument('-l', '--logfile',
                        help='Destination of error and info messages; default: stdout.',
                        dest='logfile',
                        default='stdout');
    parser.add_argument('result_file',
                        help="path to one of the result files of the Bert run; others will be derived from it")

    args = parser.parse_args();

    #***********
    #args.result_file = "/Users/paepcke/EclipseWorkspacesNew/facebook_ad_classifier/src/classifier/datasets/facebook_ads_clean_train_test_stats.dict"
    #***********


    BertResultAnalyzer(args.result_file)
    
    input("Press ENTER to close the figures and exit...")

         

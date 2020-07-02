#!/usr/bin/env python3
'''
Created on Jun 18, 2020

@author: paepcke
'''

from _collections import OrderedDict
import argparse
import csv
import json
import os
import re
import sqlite3
import sys

from sklearn.metrics import classification_report, confusion_matrix
import torch

from logging_service import LoggingService
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BertResultAnalyzer(object):
    '''
    classdocs
    '''


    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, result_file, charts=False):
        '''
        Constructor
        '''
        self.log = LoggingService()
        
        res_files_dict = self.get_result_file_paths(result_file)
        
        # Get the stats dict from disk:

        try:
            # Load the train/validate/test stats dict
            # from the db:
            stats_file = res_files_dict['stats_file']
            with open(stats_file, 'rb') as fd:
                # Load the data struct into cpu RAM, even
                # if it was on a GPU when it was saved:
                train_test_stats = json.load(fd)
                
        except FileNotFoundError:
            self.log.err(f"No train/validate/test stats file found: {stats_file}; quitting")
            sys.exit(1)

        # Get the predictions made for the testset, 
        # and the corresponding true labels
        
        # The file is simple csv:
        #     prediction,true_label
        #        0      ,   1
        #        2      ,   2
        #         ...
        # For robustness, don't make assumption about
        # the column order, but read it from the first
        # row. We build a two-key dict: {'prediction' : [...],
        #                                'true_label' : [...]
        #                               }
        # The most pythonic method would be to read all
        # the rows, forming an array of tuples, and then
        # use the star operator (*). But I have to look
        # up the semantics of that op every time. So, no.
        try:
            preds_file = res_files_dict['preds_file']
            with open(preds_file, 'r') as fd:
                reader = csv.reader(fd)
                col_names = next(reader)
                pred_vs_true_dict = {col_names[0] : [],
                                     col_names[1] : []
                                     }
                for pred_vs_true in reader:
                    pred_vs_true_dict[col_names[0]].append(int(pred_vs_true[0]))
                    pred_vs_true_dict[col_names[1]].append(int(pred_vs_true[1]))
        except FileNotFoundError:
            self.log.err(f"No test predictions file found ({res_files_dict['preds_file']})")
            sys.exit(1)

        # Print descriptives:
        db_file = res_files_dict['db_file']
        try:
            self.db = sqlite3.connect(db_file)
            self.db.row_factory = sqlite3.Row
            
            # Get ordered dict mapping int labels to
            # text labels:
            
            self.label_encodings = self.get_label_encodings()

            self.get_descriptives(train_test_stats, pred_vs_true_dict)
     
            # Plot train/val losses by epoch:
            if charts:
                self.plot_train_val_loss_and_accuracy(train_test_stats)
        finally:
            self.db.close()

    #------------------------------------
    # get_descriptives
    #-------------------
    
    def get_descriptives(self, train_test_stats, prediction_vs_labels_dict): 

        '''
        Given a dict like the following, which was stored
        by the training process by the self.db Sqlite db,
        get more result info from that db, and print result
        evaluations:
        
            training_stats:
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
        
        
        @param train_test_stats: dict of test and training results
        @type train_test_stats: dict
        @param prediction_vs_labels_dict: dict with keys 'prediction',
            and 'true_labels'.
        @type prediction_vs_labels_dict: {str:[int]}
        '''

        # Convenience: pull out the Testing sub-dir:
        test_res_dict = train_test_stats['Testing']

        # Get distribution of labels across the entire dataset,
        # and the train, validation, and test sets:
        
         
        # Get overall label distribution:
        
        res = self.db.execute('''SELECT label, count(*) AS num_this_label 
                            FROM Samples 
                           GROUP BY label;
                        ''')
        label_count_dict_whole_set = {}
        # Build dict: string-label ===> number of samples
        for (int_label, num_this_label) in res:
            # Get str label from int label:
            str_label = self.label_encodings[int_label]
            label_count_dict_whole_set[str_label] = num_this_label
            
        # Get train set label distribution:
        
        res = self.db.execute('''SELECT label, count(*) as label_count
                              FROM TrainQueue LEFT JOIN Samples
                               ON TrainQueue.sample_id = Samples.sample_id
                             GROUP BY label;
                        ''')
        label_count_dict_train = {}
        # Build dict: string-label ===> number of samples
        for (int_label, num_this_label) in res:
            # Get str label from int label:
            str_label = self.label_encodings[int_label]
            label_count_dict_train[str_label] = num_this_label

        # Get validation set label distribution:

        res = self.db.execute('''SELECT label, count(*) as label_count
                              FROM ValidateQueue LEFT JOIN Samples
                               ON ValidateQueue.sample_id = Samples.sample_id
                             GROUP BY label;
                        ''')
        
        label_count_dict_validate = {}
        # Build dict: string-label ===> number of samples
        for (int_label, num_this_label) in res:
            # Get str label from int label:
            str_label = self.label_encodings[int_label]
            label_count_dict_validate[str_label] = num_this_label

        # Get test set label distribution:
        
        res = self.db.execute('''SELECT label, count(*) as label_count
                              FROM TestQueue LEFT JOIN Samples
                               ON TestQueue.sample_id = Samples.sample_id
                             GROUP BY label;
                        ''')
        
        label_count_dict_test = {}
        # Build dict: string-label ===> number of samples
        for (int_label, num_this_label) in res:
            # Get str label from int label:
            str_label = self.label_encodings[int_label]
            label_count_dict_test[str_label] = num_this_label

        # Put the all test results into 
        # a dataframe for easy printing:
        train_res_df = pd.DataFrame(test_res_dict,
                                    index=[0])
        # Same for label value distributions:
        samples_label_distrib_df    = pd.DataFrame(label_count_dict_whole_set,
                                                   index=[0]
                                                   )
        train_label_distrib_df      = pd.DataFrame(label_count_dict_train,
                                                   index=[0]
                                                   )
        validate_label_distrib_df   = pd.DataFrame(label_count_dict_validate,
                                                   index=[0]
                                                   )
        
        test_label_distrib_df   = pd.DataFrame(label_count_dict_test,
                                               index=[0]
                                               )
        
        # Turn confusion matrix numpy into a df
        # with string labels to mark rows and columns:
        true_labels = prediction_vs_labels_dict['true_label']
        test_predictions = prediction_vs_labels_dict['prediction']
        int_labels = list(self.label_encodings.keys())
        str_labels = list(self.label_encodings.values())
        conf_mat_df = pd.DataFrame(confusion_matrix(true_labels,
                                                    test_predictions,
                                                    labels=int_labels
                                                    ),
                                   index=str_labels,
                                   columns=str_labels
                                   )
        # We also produce a conf matrix normalized to 
        # the true values. So each cell is percentage 
        # predicted/true:
        conf_mat_norm_df = pd.DataFrame(confusion_matrix(true_labels,
                                                         test_predictions,
                                                         normalize='true',
                                                         labels=int_labels
                                                         ),
                                                         index=str_labels,
                                                         columns=str_labels
                                        )
        # Change entries to be 'x.yy%'
        #conf_mat_norm_df = conf_mat_norm_df.applymap(lambda df_el: f"{round(df_el,2)}%")
        conf_mat_norm_df = conf_mat_norm_df.applymap(lambda df_el: f"{100*round(df_el,1)}%")
        
        print(train_res_df.to_string(index=False, justify='center'))
        print()
        # Label distributions in the sample subsets:
        print("Distribution of labels across all samples:")
        print(samples_label_distrib_df.to_string(index=False, justify='center'))
        print("Distribution of labels across training set:")
        print(train_label_distrib_df.to_string(index=False, justify='center'))
        print("Distribution of labels across validation set:")
        print(validate_label_distrib_df.to_string(index=False, justify='center'))
        print("Distribution of labels across test set:")
        print(test_label_distrib_df.to_string(index=False, justify='center'))
        print()
        print(f"Confusion matrix (rows: true; cols: predicted):")
        print(f"{conf_mat_df}")
        print("")
        print(f"Confusion matrix normalized: percent of true (rows: true; cols: predicted):")
        print(conf_mat_norm_df)
        print("")
        result_report = classification_report(true_labels,
                                              test_predictions)
        print(result_report)

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
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_xticks(df_stats.index.values)

        ax1.legend(frameon=False)
        ax2.legend(frameon=False)
        
        plt.ion()
        plt.show(block=False)

    #------------------------------------
    # get_result_file_paths 
    #-------------------
    
    def get_result_file_paths(self, one_result_file):
        '''
        Given one of the result files that are produced
        by bert_train_parallel.py, return a dict with the
        full paths of them all. For example, given path
        /foo/bar/facebook_ads.sqlite, constructs path to
        
        
           /foo/bar/facebook_ads_clean_testset_predictions.csv
           /foo/bar/facebook_ads_clean_train_test_stats.json
           /foo/bar/facebook_ads_clean_trained_model.sav
           /foo/bar/facebook_ads_clean.sqlite
           /foo/bar/facebook_ads_clean.csv
           
        Returns:
                'preds_file' : file_path,
                'stats_file' : file_path,
                'model_file' : file_path,
                'db_file'    : file_path
                }

        @param one_result_file:
        @type one_result_file:
        '''
        preds_str = '_testset_predictions.csv'
        stats_str = '_train_test_stats.json'
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
    # get_label_encodings 
    #-------------------
    
    def get_label_encodings(self):
        '''
        Get contents of the LabelEncodings table
        into the dict label_encodings. These are
        the mappings from the bert integer encodings
        of labels (0,1,2,3) to the human readable labels
        ('foo', 'bar', 'fum', 'blue') 
        '''
        label_encodings = OrderedDict()
        try:
            cur = self.db.execute(f'''SELECT key_col, val_col 
                                        FROM LabelEncodings''')
            while True:
                (int_label, str_label) = next(cur)
                label_encodings[int(int_label)] = str_label
        except StopIteration:
            return label_encodings
                             


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
    parser.add_argument('-c', '--charts',
                        action='store_true',
                        default=False,
                        help="If set various result charts are drawn."
                        )                        
    parser.add_argument('result_file',
                        help="path to one of the result files of the Bert run; others will be derived from it")

    args = parser.parse_args();

    #***********
    #args.result_file = "/Users/paepcke/EclipseWorkspacesNew/facebook_ad_classifier/src/classifier/datasets/facebook_ads_clean_train_test_stats.dict"
    #***********


    BertResultAnalyzer(args.result_file, charts=args.charts)
    
    if args.charts:
        input("Press ENTER to close the figures and exit...")

         

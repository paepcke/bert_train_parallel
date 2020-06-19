#!/usr/bin/env python3
'''
Created on Jun 18, 2020

@author: paepcke
'''

import argparse
import os
import re
import sys

import torch

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

    def __init__(self, result_file):
        '''
        Constructor
        '''
        res_files_dict = self.get_result_file_paths(result_file)
        
        # Plot train/val losses by epoch:
        
        with open(res_files_dict['stats_file'], 'rb') as fd:
            # Load the data struct into cpu RAM, even
            # if it was on a GPU when it was saved:
            train_test_stats = torch.load(fd,
                                          map_location=torch.device('cpu')
                                          )
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
        
        self.plot_train_val_loss_and_accuracy(train_test_stats)

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
        epoch_stats_dicts.extend(
            [
                {'epoch': 2, 'Training Loss': 0.01250069046020508, 'Validation Loss': 0.0623801279067993, 'Training Accuracy': 0.01500625, 'Validation Accuracy.': 0.11, 'Training Time': '0:00:24', 'Validation Time': '0:00:01'},
                {'epoch': 3, 'Training Loss': 0.00250069046020508, 'Validation Loss': 0.0323801279067993, 'Training Accuracy': 0.02500625, 'Validation Accuracy.': 0.25, 'Training Time': '0:00:24', 'Validation Time': '0:00:01'},
                {'epoch': 4, 'Training Loss': 0.0004069046020508, 'Validation Loss': 0.0023801279067993, 'Training Accuracy': 0.01500625, 'Validation Accuracy.': 0.35, 'Training Time': '0:00:24', 'Validation Time': '0:00:01'}
            ]
            )
        #********

        self.plot_stats_dataframe(epoch_stats_dicts, 'loss')
        self.plot_stats_dataframe(epoch_stats_dicts, 'accuracy')

    #------------------------------------
    # def plot_stats_dataframe 
    #-------------------

    def plot_stats_dataframe(self, epoch_stats_dicts, plot_type):
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
        plt.rcParams["figure.figsize"] = (12,6)
        
        # Plot the learning curve.
        if plot_type == 'loss':
            plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
            plt.plot(df_stats['Validation Loss'], 'g-o', label="Validation")
            # Label the plot.
            plt.title("Training & Validation Loss")
            plt.ylabel("Loss")
        elif plot_type == 'accuracy':
            plt.plot(df_stats['Training Accuracy'], 'b-o', label="Training")
            plt.plot(df_stats['Validation Accuracy'], 'g-o', label="Validation")
            # Label the plot.
            plt.title("Training & Validation Accuracy")
            plt.ylabel("Accuracy")

        plt.xlabel("Epoch")
        plt.legend()
        plt.xticks(df_stats.index.values)
        #plt.xticks([1, 2, 3, 4])
        
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

    #------------------------------------
    # print_test_results 
    #-------------------
    
    def print_test_results(self, training_stats_info=None):
        
        if training_stats_info is None:
            training_stats_info = self.training_stats
        #self.log.info(self.training_stats)
        self.plot_train_val_loss_and_accuracy(self.training_stats)
        return
        #***********
#         test_count = 0
#         unsure_count = 0
#         count = 0
#         neutral_count = 0
#         neutral_test = 0
#         left_test = 0
#         left_count = 0
#         right_test = 0
#         right_count = 0
        
#         for i in range(len(self.dataloader)):
#             y_label = flat_predictions[i]
#             category = flat_true_labels[i]
#             count += 1
#             if (category == 2):
#                 neutral_count += 1
#             if (category == 1):
#                 left_count += 1
#             if (category == 0):
#                 right_count += 1
#             if (y_label == category):
#                 test_count += 1
#                 if (category == 2):
#                     neutral_count += 1
#                 if (category == 0):
#                     right_test += 1
#                 if (category == 1):
#                     left_test += 1
#                 # print("CORRECT!")
#                 # print(df['message'][i], y_label)
#                 # print("is : ", category)
#             else:
#                 # print("WRONG!")
#                 # print(df['message'][i], y_label)
#                 # print("is actually: ", category)
#                 # print(test_count, "+", unsure_count, "out of", count)
#                 pass
#         print("neutral: ", neutral_test, "/", neutral_count)
#         print("left: ", left_test, "/", left_count)
#         print("right: ", right_test, "/", right_count)
#         print(test_count, "+", unsure_count, "out of", count)
#         
#         print(accuracy_score(flat_true_labels, flat_predictions))
#                 
#         # Format confusion matrix:
#             
#         #             right   left    neutral
#         #     right
#         #     left
#         #     neutral
#         
#         results = confusion_matrix(flat_true_labels, flat_predictions) 
#           
#         print('Confusion Matrix :')
#         print(results) 
#         print('Accuracy Score :',accuracy_score(flat_true_labels, flat_predictions))
#         print('Report : ')
#         print(classification_report(flat_true_labels, flat_predictions))

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

         

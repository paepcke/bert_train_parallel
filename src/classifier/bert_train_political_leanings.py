'''
Created on Jun 8, 2020

@author: paepcke
'''
#**********
#import sys;sys.path.append(r'/Users/paepcke/.p2/pool/plugins/org.python.pydev.core_7.5.0.202001101138/pysrc')
#import pydevd;pydevd.settrace()
#**********    


from transformers import AdamW, BertForSequenceClassification

import os,sys
import ast
import re
import time
import torch
from torch import nn, cuda
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
#from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from bert_training.bert_fine_tuning_sentence_classification import df
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns

class PoliticalLeaningsAnalyst(object):
    '''
    For this task, we first want to modify the pre-trained 
    BERT model to give outputs for classification, and then
    we want to continue training the model on our dataset
    until that the entire model, end-to-end, is well-suited
    for our task. 
    
    Thankfully, the huggingface pytorch implementation 
    includes a set of interfaces designed for a variety
    of NLP tasks. Though these interfaces are all built 
    on top of a trained BERT model, each has different top 
    layers and output types designed to accomodate their specific 
    NLP task.  
    
    Here is the current list of classes provided for fine-tuning:
    * BertModel
    * BertForPreTraining
    * BertForMaskedLM
    * BertForNextSentencePrediction
    * **BertForSequenceClassification** - The one we'll use.
    * BertForTokenClassification
    * BertForQuestionAnswering
    
    The documentation for these can be found under 
    https://huggingface.co/transformers/v2.2.0/model_doc/bert.html
    
    '''
    SPACE_TO_COMMA_PAT = re.compile(r'([0-9])[\s]+')
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 model_save_path,
                 epochs=4,
                 batch_size=8,
                 max_seq_len=128,
                 learning_rate=3e-5
                 ):
        '''
        Number of epochs: 2, 3, 4 
        '''
        self.batch_size = self.batch_size
        gpu_device = self.enable_GPU()

        # TRAINING:        
        train_set_loader = self.load_dataset("leanings_right_sized.csv")
        test_set_loader  = self.load_dataset("leanings_right_sized_testset.csv")
        
        (train_labels, train_input_ids, train_attention_masks) = self.init_label_info(train_set_loader)
        (train_dataloader, validation_dataloader) = \
           self.prepare_input_stream(train_input_ids, 
                                     train_labels, 
                                     train_attention_masks, 
                                     batch_size)
        (model, train_optimizer, train_scheduler) = self.prepare_model(train_dataloader, 
                                                                       learning_rate)
        self.train(model, 
                   train_dataloader, 
                   validation_dataloader, 
                   train_optimizer, 
                   train_scheduler, 
                   epochs, 
                   gpu_device)
        
        # TESTING:
        # Have test set loader, need input_ids and labels
        (test_labels, test_input_ids, test_attention_masks) = self.init_label_info(test_set_loader)
        (test_dataloader, _test_validation_dataloader) = \
           self.prepare_input_stream(test_input_ids, 
                                     test_labels, 
                                     test_attention_masks, 
                                     batch_size)
        self.test(model, test_input_ids, test_dataloader, gpu_device)
        
        # EVALUATE RESULT:
        
        self.compute_matthews_coefficient(model, test_dataloader, gpu_device)
        
        # Save the model on the VM
        print(f"Saving model to VM...")
        torch.save(model,open('curr_model.sav', 'wb'))
        
        print("Copying model to Google Drive")

#       Note: To maximize the score, we should remove the "validation set", 
#       which we used to help determine how many epochs to train for, and 
#       train on the entire training set.

    #------------------------------------
    # enable_GPU 
    #-------------------

    def enable_GPU(self):
        # Get the GPU device name.
        device_name = tf.test.gpu_device_name()

        # The device name should look like the following:
        if device_name == '/device:GPU:0':
            print('Found GPU at: {}'.format(device_name))
        else:
            raise SystemError('GPU device not found')
        # From torch:
        device = cuda.current_device()
        return device

    #------------------------------------
    # load_dataset 
    #-------------------

    def load_dataset(self, path):

        # The Pandas.to_csv() method writes numeric Series 
        # as a string: "[ 10   20   30]", so need to replace
        # the white space with commas. Done via the following
        # conversion function:
        
        # Find a digit followed by at least one whitespace: space or
        # newline. Remember the digit as a capture group: the parens:
        

        
        df = pd.read_csv(path,
                         delimiter=',', 
                         header=0, 
                         converters={'ids' : self.to_np_array}
                        )
        self.train_set = df
        (labels, input_ids, attention_masks) = self.init_label_info(df)

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        labels = torch.tensor(labels)

        data = TensorDataset(input_ids, attention_masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, 
                                sampler=sampler, 
                                batch_size=self.batch_size)
        return dataloader

    #------------------------------------
    # init_label_info 
    #-------------------
    
    def init_label_info(self, df):
        '''
        Extract labels and input_ids of a dataset.
        Compute attention masks.
        
        @param df: data frame with at least columns
            label, tokens, input_ids
        @type df: DataFrame
        @return: (labels, input_ids, attention_masks)
        '''
        
        labels = df.leaning.values
        # Labels must be int-encoded:
        for i in range(len(labels)):
            if labels[i] == 'right':
                labels[i] = 0
            if labels[i] == 'left':
                labels[i] = 1
            if labels[i] == 'neutral':
                labels[i] = 2
        
        # Extract the sentences and labels of our training 
        # set as numpy ndarrays.
        
        labels = self.train_set.leaning.values
        # Labels must be int-encoded:
        for i in range(len(labels)):
            if labels[i] == 'right':
                labels[i] = 0
            if labels[i] == 'left':
                labels[i] = 1
            if labels[i] == 'neutral':
                labels[i] = 2
        
        # Grab the BERT index ints version of the tokens:
        input_ids = self.train_set.ids
        
        # Create attention masks
        attention_masks = []
        
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        
        return (labels, input_ids, attention_masks)

    #------------------------------------
    # prepare_input_stream 
    #-------------------

    def prepare_input_stream(self, input_ids, labels, attention_masks, batch_size):
        '''

        Divide up our training set to use 90% for training 
        and 10% for validation.
        
        We'll also create an iterator for our dataset 
        using the torch DataLoader class. This helps 
        save on memory during training because, unlike 
        a for loop, with an iterator the entire dataset 
        does not need to be loaded into memory.
        
        @param input_ids: array of BERT vocal indices
        @type input_ids: nparray
        @param labels: array of label identifiers, coded as ints
        @type labels: int
        @param attention_masks: mask over tokens to indicate 
            padding (0s) from real tokens (1s)
        @type attention_masks: nparray
        @param batch_size: number of input records to process
            at a time
        @type batch_size: int
        @return: train dataloader
        @return: validation dataloader
        @rtype: DataLoader
        '''
        # Use train_test_split to split our data into train and validation sets for training
        
        (train_inputs, validation_inputs, 
        train_labels, validaton_labels) = train_test_split(input_ids, labels, 
                                                           random_state=2018, test_size=0.1)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                               random_state=2018, test_size=0.1)

        # Convert all of our data into torch tensors, 
        # the required datatype for our model
        
        train_inputs = torch.tensor(list(train_inputs))
        validation_inputs = torch.tensor(list(validation_inputs))
        
        train_labels = torch.tensor(train_labels)
        validaton_labels = torch.tensor(validaton_labels)
        
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, 
                                      sampler=train_sampler, 
                                      batch_size=batch_size)
        
        validation_data = TensorDataset(validation_inputs, validation_masks, validaton_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
        
        return (train_dataloader, validation_dataloader)

    #------------------------------------
    # prepare_model 
    #-------------------

    def prepare_model(self, train_dataloader, learning_rate):
        '''
        - Batch size: no more than 8 for a 16GB GPU. Else 16, 32  
        - Learning rate (for the Adam optimizer): 5e-5, 3e-5, 2e-5  

        The epsilon parameter `eps = 1e-8` is "a very small 
        number to prevent any division by zero in the 
        implementation
        
        @param train_dataloader: data loader for model input data
        @type train_dataloader: DataLoader
        @param learning_rate: amount of weight modifications per cycle.
        @type learning_rate: float
        @return: model
        @rtype: BERT pretrained model
        @return: optimizer
        @rtype: Adam
        @return: scheduler
        @rtype: Schedule (?)
        '''
        
        # Load BertForSequenceClassification, the pretrained BERT model with a single 
        # linear classification layer on top. 
        model = BertForSequenceClassification.from_pretrained(
            #"bert-large-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            "bert-base-uncased",
            num_labels = 2, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether th`e model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        
        # Tell pytorch to run this model on the GPU.
        model.cuda()
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          #lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          lr = learning_rate,
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )
        
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * self.epochs
        
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
        return (model, optimizer, scheduler)

    #------------------------------------
    # train 
    #-------------------

    def train(self, 
              model, 
              train_dataloader,
              validation_dataloader, 
              optimizer, 
              scheduler, 
              epochs, 
              gpu_device):
        '''
        Below is our training loop. There's a lot going on, but fundamentally 
        for each pass in our loop we have a trianing phase and a validation phase. 
        
        **Training:**
        - Unpack our data inputs and labels
        - Load data onto the GPU for acceleration
        - Clear out the gradients calculated in the previous pass. 
            - In pytorch the gradients accumulate by default 
              (useful for things like RNNs) unless you explicitly clear them out.
        - Forward pass (feed input data through the network)
        - Backward pass (backpropagation)
        - Tell the network to update parameters with optimizer.step()
        - Track variables for monitoring progress
        
        **Evalution:**
        - Unpack our data inputs and labels
        - Load data onto the GPU for acceleration
        - Forward pass (feed input data through the network)
        - Compute loss on our validation data and track variables for monitoring progress
        
        Pytorch hides all of the detailed calculations from us, 
        but we've commented the code to point out which of the 
        above steps are happening on each line. 
        
        PyTorch also has some 
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) which you may also find helpful.*
        '''
        datetime.datetime.now().isoformat()
        
        # This training code is based on the `run_glue.py` script here:
        # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
        
        # Set the seed value all over the place to make this reproducible.
        seed_val = 42
        
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        # From torch:
        cuda.manual_seed_all(seed_val)
        
        # We'll store a number of quantities such as training and validation loss, 
        # validation accuracy, and timings.
        training_stats = []
        
        # Measure the total training time for the whole run.
        total_t0 = time.time()
        
        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            # Perform one full pass over the training set.
        
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
        
            # Measure how long the training epoch takes.
            t0 = time.time()
        
            # Reset the total loss for this epoch.
            total_train_loss = 0
        
            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()
        
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):
        
                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.format_time(time.time() - t0)
                    
                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(gpu_device)
                b_input_mask = batch[1].to(gpu_device)
                b_labels = batch[2].to(gpu_device)
        
                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because 
                # accumulating the gradients is "convenient while training RNNs". 
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                model.zero_grad()        
        
                # Perform a forward pass (evaluate the model on this training batch).
                # The documentation for this `model` function is here: 
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # It returns different numbers of parameters depending on what arguments
                # arge given and what flags are set. For our useage here, it returns
                # the loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                loss, logits = model(b_input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=b_input_mask, 
                                     labels=b_labels)
        
                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += loss.item()
        
                # Perform a backward pass to calculate the gradients.
                loss.backward()
        
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                # From torch:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()
        
                # Update the learning rate.
                scheduler.step()
        
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader)            
            
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)
        
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
        
            print("")
            print("Running Validation...")
        
            t0 = time.time()
        
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()
        
            # Tracking variables 
            total_eval_accuracy = 0
            total_eval_loss = 0
            #nb_eval_steps = 0
        
            # Evaluate data for one epoch
            for batch in validation_dataloader:
                
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels 
                b_input_ids = batch[0].to(gpu_device)
                b_input_mask = batch[1].to(gpu_device)
                b_labels = batch[2].to(gpu_device)
                
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        
        
                    # Forward pass, calculate logit predictions.
                    # token_type_ids is the same as the "segment ids", which 
                    # differentiates sentence 1 and 2 in 2-sentence tasks.
                    # The documentation for this `model` function is here: 
                    # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                    # Get the "logits" output by the model. The "logits" are the output
                    # values prior to applying an activation function like the softmax.
                    (loss, logits) = model(b_input_ids, 
                                           token_type_ids=None, 
                                           attention_mask=b_input_mask,
                                           labels=b_labels)
                    
                # Accumulate the validation loss.
                total_eval_loss += loss.item()
        
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
        
                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)
                
        
            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            
            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)
            
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            print("  Validation took: {:}".format(validation_time))
        
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        print("")
        print("Training complete!")
        
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        
        datetime.datetime.now().isoformat()

    #------------------------------------
    # test 
    #-------------------

    def test(self, model, input_ids, prediction_dataloader, gpu_device):
        '''
        pply our fine-tuned model to generate predictions on the test set.
        '''
        
        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))
        
        # Put model in evaluation mode
        model.eval()
        
        # Tracking variables 
        predictions , true_labels = [], []
        
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(gpu_device) for t in batch)
            
            #***** Are the following correct?
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
            
            logits = outputs[0]
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)
        
        print('    DONE.')
        return(true_labels)

    #------------------------------------
    # compute_matthews_coefficient 
    #-------------------
    
    def compute_matthews_coefficient(self, model, prediction_dataloader, gpu_device):
        # Tracking variables 
        predictions , true_labels = [], []
        
        # Predict 
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(gpu_device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
            # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
          
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)
        
        del batch
        del logits
        # From torch:
        cuda.empty_cache()
        
        # Combine the results for all of the batches and calculate our final MCC score.
        
        # Combine the results across all batches. 
        flat_predictions = np.concatenate(predictions, axis=0)
        
        # For each sample, pick the label (0 or 1) with the higher score.
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        
        # Combine the correct labels for each batch into a single list.
        flat_true_labels = np.concatenate(true_labels, axis=0)
        
        # Calculate the MCC
        mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
        
        print('Total MCC: %.3f' % mcc)
        return mcc

    #------------------------------------
    # print_test_results 
    #-------------------
    
    def print_test_results(self, predictions, true_labels):
        # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        print(flat_predictions)
        print(flat_true_labels)
        
        test_count = 0
        unsure_count = 0
        count = 0
        neutral_count = 0
        neutral_test = 0
        left_test = 0
        left_count = 0
        right_test = 0
        right_count = 0
        
        for i in range(len(df)):
            y_label = flat_predictions[i]
            category = flat_true_labels[i]
            count += 1
            if (category == 2):
                neutral_count += 1
            if (category == 1):
                left_count += 1
            if (category == 0):
                right_count += 1
            if (y_label == category):
                test_count += 1
                if (category == 2):
                    neutral_count += 1
                if (category == 0):
                    right_test += 1
                if (category == 1):
                    left_test += 1
                # print("CORRECT!")
                # print(df['message'][i], y_label)
                # print("is : ", category)
            else:
                # print("WRONG!")
                # print(df['message'][i], y_label)
                # print("is actually: ", category)
                # print(test_count, "+", unsure_count, "out of", count)
                pass
        print("neutral: ", neutral_test, "/", neutral_count)
        print("left: ", left_test, "/", left_count)
        print("right: ", right_test, "/", right_count)
        print(test_count, "+", unsure_count, "out of", count)
        
        print(accuracy_score(flat_true_labels, flat_predictions))
                
        # Format confusion matrix:
            
        #             right   left    neutral
        #     right
        #     left
        #     neutral
        
        results = confusion_matrix(flat_true_labels, flat_predictions) 
          
        print('Confusion Matrix :')
        print(results) 
        print('Accuracy Score :',accuracy_score(flat_true_labels, flat_predictions))
        print('Report : ')
        print(classification_report(flat_true_labels, flat_predictions))


# ---------------------- Utilities ----------------------

    #------------------------------------
    # prepare_model_save 
    #-------------------
    
    def prepare_model_save(self, model_file):
        if os.path.exists(model_file):
            print(f"File {model_file} exists")
            print("If intent is to load it, go to cell 'Start Here...'")
            print("Else remove on google drive, or change model_file name")
            print("Removal instructions: Either execute 'os.remove(model_file)', or do it in Google Drive")
            sys.exit(1)

        # File does not exist. But ensure that all the 
        # directories on the way exist:
        paths = os.path.dirname(model_file)
        try:
            os.makedirs(paths)
        except FileExistsError:
            pass

    #------------------------------------
    # to_np_array 
    #-------------------

    def to_np_array(self, array_string):
        # Use the pattern to substitute occurrences of
        # "123   45" with "123,45". The \1 refers to the
        # digit that matched (i.e. the capture group):
        proper_array_str = PoliticalLeaningsAnalyst.SPACE_TO_COMMA_PAT.sub(r'\1,', array_string)
        # Turn from a string to array:
        return np.array(ast.literal_eval(proper_array_str))

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
        
        print('The BERT model has {:} different named parameters.\n'.format(len(params)))
        print('==== Embedding Layer ====\n')
        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== First Transformer ====\n')
        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
        print('\n==== Output Layer ====\n')
        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    #------------------------------------
    # plot_train_val_loss 
    #-------------------
    
    def plot_train_val_loss(self, training_stats):
        '''
        View the summary of the training process.
        '''
        
        # Display floats with two decimal places.
        pd.set_option('precision', 2)
        
        # Create a DataFrame from our training statistics.
        df_stats = pd.DataFrame(data=training_stats)
        ""
        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('epoch')
        
        # A hack to force the column headers to wrap.
        #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
        
        # Display the table.
        df_stats
        
        # Notice that, while the the training loss is 
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
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
        
        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])
        
        plt.show()

    #------------------------------------
    # flat_accuracy 
    #-------------------

    def flat_accuracy(self, preds, labels):
        '''
        Function to calculate the accuracy of our predictions vs labels
        '''
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    #------------------------------------
    # format_time  
    #-------------------

    def format_time(self, elapsed):
        '''
        Helper function for formatting elapsed times as `hh:mm:ss`
        Takes a time in seconds and returns a string hh:mm:ss
        '''
    
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

# -------------------- Main ----------------
if __name__ == '__main__':
    pass
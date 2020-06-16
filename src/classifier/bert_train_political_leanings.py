'''
Created on Jun 8, 2020

@author: paepcke
'''
#**********
#import sys;sys.path.append(r'/Users/paepcke/.p2/pool/plugins/org.python.pydev.core_7.5.0.202001101138/pysrc')
#import pydevd;pydevd.settrace()
#**********    



#import ast
import datetime
import os, sys
import random
import time

sys.path.append(os.path.dirname(__file__))

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import matthews_corrcoef
from torch import nn, cuda
import torch
from transformers import AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

from bert_feeder_dataloader import BertFeederDataloader
from bert_feeder_dataset import BertFeederDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bert_feeder_dataloader import set_split_id
from logging_service import LoggingService
import tensorflow as tf


sys.path.append(os.path.dirname(__file__))

#from torch.utils.data import DataLoader, SequentialSampler



#from torch.utils.data import IterableDataset
#from sklearn.metrics import accuracy_score 
#???from bert_training.bert_fine_tuning_sentence_classification import df
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
#     SPACE_TO_COMMA_PAT = re.compile(r'([0-9])[\s]+')
    
    RANDOM_SEED = 3631
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 csv_path,
                 model_save_path=None,
                 text_col_name='text',
                 label_col_name='label',
                 epochs=4,
                 batch_size=8,
                 sequence_len=128,
                 learning_rate=3e-5
                 ):
        '''
        Number of epochs: 2, 3, 4 
        '''
        
        logfile_name = os.path.join(os.path.dirname(__file__), 'facebook_train.log')
        self.log = LoggingService(logfile=logfile_name)
        self.batch_size = batch_size
        self.epochs     = epochs
        self.gpu_device = self.enable_GPU()

        if model_save_path is None:
            (file_root, _ext) = os.path.splitext(csv_path)
            model_save_path = file_root + '.sav'
        # Preparation:
        dataset = BertFeederDataset(csv_path,
                                    text_col_name=text_col_name,
                                    label_col_name=label_col_name,
                                    sequence_len=sequence_len
                                    )
                #sampler = SequentialSampler(dataset)
        dataloader = BertFeederDataloader(dataset, 
                                #sampler=sampler, 
                                batch_size=self.batch_size)

        self.dataloader = dataloader

        # Have the loader prepare 
        dataloader.split_dataset(train_percent=0.8,
                                 val_percent=0.1,
                                 test_percent=0.1,
                                 random_seed=self.RANDOM_SEED)

        # TRAINING:        
        dataset.switch_to_split('train')
        (model, train_optimizer, train_scheduler) = self.prepare_model(dataloader,
                                                                       learning_rate)
        self.train(model, 
                   dataloader,
                   train_optimizer, 
                   train_scheduler, 
                   epochs) 
        
        # TESTING:

        dataloader.switch_to_split('test')
        (prediction_res, labels) = self.test(model, dataloader)
        (prediction_logits, _something) = prediction_res
        (labels, _something) = labels
        
        # EVALUATE RESULT:
        # Calculate the MCC
        predictions = self.logits_to_classes(prediction_logits)
        labels = labels.numpy()
        mcc = self.matthews_corrcoef(predictions, labels)
        self.log.info(f"Test Matthew's coefficient: {mcc}")
        test_accuracy = self.accuracy(predictions, labels)
        self.log.info(f"Accuracy on test set: {test_accuracy}")
        
        # Save the model on the VM
        self.log.info(f"Saving model to {model_save_path} ...")
        with open(model_save_path, 'wb') as fd:
            torch.save(model, fd)
            
        (path, _ext) = os.path.splitext(model_save_path)
        predictions_path = f"{path}_predictions.np"
        self.log.info(f"Saving predictions to {predictions_path}")
        np.save(predictions_path, predictions)
        
        #print("Copying model to Google Drive")

#       Note: To maximize the score, we should remove the "validation set", 
#       which we used to help determine how many epochs to train for, and 
#       train on the entire training set.

    #------------------------------------
    # enable_GPU 
    #-------------------

    def enable_GPU(self):

        # Get the GPU device name.
        # Could be (GPU available):
        #   device(type='cuda', index=0)
        # or (No GPU available):
        #   device(type='cpu')

        device_name = tf.test.gpu_device_name()
        if len(device_name) == 0:
            return 'cpu'
        # Got something like: '/device:GPU:0
        (_device_mnt, device_type, device_indx) = device_name.split(':')
        if device_type == 'GPU':
            return device_indx
        else:
            device = 'cpu'
        return device
    
#     #------------------------------------
#     # load_dataset 
#     #-------------------
# 
#     def load_dataset(self, path):
# 
#         # The Pandas.to_csv() method writes numeric Series 
#         # as a string: "[ 10   20   30]", so need to replace
#         # the white space with commas. Done via the following
#         # conversion function:
#         
#         # Find a digit followed by at least one whitespace: space or
#         # newline. Remember the digit as a capture group: the parens:
#         
# 
#         
#         df = pd.read_csv(path,
#                          delimiter=',', 
#                          header=0, 
#                          converters={'ids' : self.to_np_array}
#                         )
#         self.train_set = df
#         (labels, input_ids, attention_masks) = self.init_label_info(df)
# 
#         input_ids = torch.tensor(input_ids)
#         attention_masks = torch.tensor(attention_masks)
#         labels = torch.tensor(labels)
# 
#         data = TensorDataset(input_ids, attention_masks, labels)
#         sampler = SequentialSampler(data)
#         dataloader = DataLoader(data, 
#                                 sampler=sampler, 
#                                 batch_size=self.batch_size)
#         return dataloader

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
        
        # Extract the sentences and labels of our training 
        # set as numpy ndarrays.
        labels = df.leaning.values
        # Labels must be int-encoded:
        label_encodings = []
        for i in range(len(labels)):
            if labels[i] == 'right':
                label_encodings.append(0)
            if labels[i] == 'left':
                label_encodings.append(1)
            if labels[i] == 'neutral':
                label_encodings.append(2)
        
        # Grab the BERT index ints version of the tokens:
        input_ids = self.train_set.ids
        
        # Create attention masks
        attention_masks = []
        
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            #seq_mask = [float(i>0) for i in seq]
            seq_mask = [int(i>0) for i in seq]
            attention_masks.append(seq_mask)
        
        return (label_encodings, input_ids, attention_masks)

#     #------------------------------------
#     # prepare_input_stream 
#     #-------------------
# 
#     def prepare_input_stream(self, input_ids, labels, attention_masks, batch_size):
#         '''
# 
#         Divide up our training set to use 90% for training 
#         and 10% for validation.
#         
#         We'll also create an iterator for our dataset 
#         using the torch DataLoader class. This helps 
#         save on memory during training because, unlike 
#         a for loop, with an iterator the entire dataset 
#         does not need to be loaded into memory.
#         
#         @param input_ids: array of BERT vocal indices
#         @type input_ids: nparray
#         @param labels: array of label identifiers, coded as ints
#         @type labels: int
#         @param attention_masks: mask over tokens to indicate 
#             padding (0s) from real tokens (1s)
#         @type attention_masks: nparray
#         @param batch_size: number of input records to process
#             at a time
#         @type batch_size: int
#         @return: train dataloader
#         @return: validation dataloader
#         @rtype: DataLoader
#         '''
#         # Use train_test_split to split our data into train and validation sets for training
#         
#         (train_inputs, validation_inputs, 
#         train_labels, validaton_labels) = train_test_split(input_ids, labels, 
#                                                            random_state=2018, test_size=0.1)
#         train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
#                                                                random_state=2018, test_size=0.1)
# 
#         # Convert all of our data into torch tensors, 
#         # the required datatype for our model
#         
#         train_inputs = torch.tensor(list(train_inputs))
#         validation_inputs = torch.tensor(list(validation_inputs))
#         
#         train_labels = torch.tensor(train_labels)
#         validaton_labels = torch.tensor(validaton_labels)
#         
#         train_masks = torch.tensor(train_masks)
#         validation_masks = torch.tensor(validation_masks)
# 
#         train_data = TensorDataset(train_inputs, train_masks, train_labels)
#         train_sampler = RandomSampler(train_data)
#         train_dataloader = DataLoader(train_data, 
#                                       sampler=train_sampler, 
#                                       batch_size=batch_size)
#         
#         validation_data = TensorDataset(validation_inputs, validation_masks, validaton_labels)
#         validation_sampler = SequentialSampler(validation_data)
#         validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
#         
#         return (train_dataloader, validation_dataloader)

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
            num_labels = 3, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        
        # Tell pytorch to run this model on the GPU.
        if self.gpu_device != 'cpu':
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
              dataloader,
              optimizer, 
              scheduler, 
              epochs): 
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
        - Load data onto the GPU for acceleration (if available)
        - Forward pass (feed input data through the network)
        - Compute train_loss on our validation data and track variables for monitoring progress
        
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
        seed_val = self.RANDOM_SEED
        
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        # From torch:
        cuda.manual_seed_all(seed_val)
        
        # We'll store a number of quantities such as training and validation train_loss, 
        # validation accuracy, and timings.
        self.training_stats = {'Training' : []}
        
        # Measure the total training time for the whole run.
        total_t0 = time.time()
        
        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            # Perform one full pass over the training set.
        
            self.log.info("")
            self.log.info(f'======== Epoch :{epoch_i + 1} / {epochs} ========')
            self.log.info('Training...')
        
            # Measure how long the training epoch takes.
            t0 = time.time()
        
            # Reset the total train_loss for this epoch.
            total_train_loss = 0.0
            total_train_accuracy = 0.0
        
            # Put the model into training mode. Don't be mislead--the call to 
            # `train` just changes the *mode*, it doesn't *perform* the training.
            # `dropout` and `batchnorm` layers behave differently during training
            # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
            model.train()
        
            # For each batch of training data...
            # Tell data loader to pull from the train sample queue:
            dataloader.switch_to_split('train')
            for step, batch in enumerate(dataloader):
        
                # Progress update every 50 batches.
                if step % 50 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = self.log.info(f"{self.format_time(time.time() - t0)}") 
                    
                    # Report progress.
                    self.log.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(dataloader), elapsed))
        
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels
                if self.gpu_device == 'cpu':
                    b_input_ids = batch['tok_ids']
                    b_input_mask = batch['attention_mask']
                    b_labels = batch['label']
                else:
                    b_input_ids = batch[0].to(self.gpu_device)
                    b_input_mask = batch[1].to(self.gpu_device)
                    b_labels = batch[2].to(self.gpu_device)
        
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
                # the train_loss (because we provided labels) and the "logits"--the model
                # outputs prior to activation.
                train_loss, logits = model(b_input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=b_input_mask, 
                                     labels=b_labels)
                
                train_acc = self.accuracy(logits, b_labels)
                total_train_accuracy += train_acc

                if self.gpu_device != 'cpu':
                    del b_input_ids
                    del b_input_mask
                    del b_labels
                    cuda.empty_cache()

                # Accumulate the training train_loss over all of the batches so that we can
                # calculate the average train_loss at the end. `train_loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value 
                # from the tensor.
                total_train_loss += train_loss.item()
        
                # Perform a backward pass to calculate the gradients.
                train_loss.backward()
        
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
        
            # Calculate the average train_loss over all of the batches.
            avg_train_loss = total_train_loss / len(dataloader)
            avg_train_accuracy = total_train_accuracy / len(dataloader)
            
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)
        
            self.log.info("")
            self.log.info(f"  Average training loss: {avg_train_loss:.2f}")
            self.log.info(f"  Average training accuracy: {avg_train_accuracy:.2f}")
            self.log.info(f"  Training epoch took: {training_time}")
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
        
            self.log.info("")
            self.log.info("Running Validation...")
        
            t0 = time.time()
        
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()
        
            # Tracking variables 
            total_val_accuracy = 0
            total_val_loss = 0
            #nb_eval_steps = 0
        
            dataloader.switch_to_split('validate')
            # Start feeding validation set from the beginning:
            dataloader.reset_split('validate')
            # Evaluate data for one epoch
            for batch in dataloader:
                
                # Unpack this training batch from our dataloader. 
                #
                # As we unpack the batch, we'll also copy each tensor to the GPU using 
                # the `to` method.
                #
                # `batch` contains three pytorch tensors:
                #   [0]: input ids 
                #   [1]: attention masks
                #   [2]: labels
                if self.gpu_device == 'cpu':
                    b_input_ids = batch['tok_ids']
                    b_input_mask = batch['attention_mask']
                    b_labels = batch['label']
                else:
                    b_input_ids = batch['tok_ids'].to(self.gpu_device)
                    b_input_mask = batch['attention_mask'].to(self.gpu_device)
                    b_labels = batch['label'].to(self.gpu_device)
                
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
                    (val_loss, logits) = model(b_input_ids, 
                                               token_type_ids=None, 
                                               attention_mask=b_input_mask,
                                               labels=b_labels)
                    
                # Accumulate the validation loss and accuracy
                total_val_loss += val_loss.item()
                total_val_accuracy += self.accuracy(logits, b_labels)
        
                # Move logits and labels to CPU
                if self.gpu_device != 'cpu':
                    logits = logits.detach().cpu().numpy()
                    _label_ids = b_labels.to('cpu').numpy()
                else:
                    _label_ids = b_labels
                    
            # Calculate the average loss over all of the batches.
            with set_split_id(dataloader, 'validate'):
                avg_val_loss = total_val_loss / len(dataloader)
                avg_val_accuracy = total_val_accuracy / len(dataloader)
            
            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)
            
            self.log.info(f"  Avg validation loss: {avg_val_loss:.2f}")
            self.log.info(f"  Avg validation accuracy: {avg_val_accuracy:.2f}")
            self.log.info(f"  Validation took: {validation_time}")
        
            # Record all statistics from this epoch.
            self.training_stats['Training'].append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Validation Loss': avg_val_loss,
                    'Training Accuracy': avg_train_accuracy,
                    'Validation Accuracy.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        self.log.info("")
        self.log.info("Training complete!")
        
        self.log.info("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        
        datetime.datetime.now().isoformat()

    #------------------------------------
    # test 
    #-------------------

    def test(self, model, dataloader):
        '''
        Apply our fine-tuned model to generate predictions on the test set.
        '''
        
        # Put model in evaluation mode
        model.eval()
        
        # Tracking variables 
        predictions , true_labels = [], []
        
        # Predict
        # Batches come as dicts with keys
        # sample_id, tok_ids, label, attention_mask: 
        for batch in dataloader:
            if self.gpu_device == 'cpu':
                b_input_ids = batch['tok_ids']
                b_input_mask = batch['attention_mask']
                b_labels = batch['label']
            else:
                b_input_ids = batch['tok_ids'].to(self.gpu_device)
                b_input_mask = batch['attention_mask'].to(self.gpu_device)
                b_labels = batch['label'].to(self.gpu_device)
            
            # Telling the model not to compute or store gradients, saving memory and 
            # speeding up prediction
            with torch.no_grad():
                (loss, logits) = model(b_input_ids, 
                                       token_type_ids=None, 
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                    
            # Move logits and labels to CPU, if the
            # are not already:
            if self.gpu_device != 'cpu':
                logits = logits.detach().cpu().numpy()
                b_labels = batch['label'].to('cpu').numpy()
            
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(b_labels)
        
        self.log.info('    DONE applying model to test set.')
        self.training_stats['Testing'] = \
                {
                    'Test Loss': loss,
                    'Test Accuracy': self.accuracy(logits, b_labels),
                    'Matthews corrcoef': self.matthews_corrcoef(logits, b_labels),
                    'Confusion matrix' : self.confusion_matrix(logits, b_labels)
                }
                
        return(predictions, true_labels)


    #------------------------------------
    # matthews_corrcoef
    #-------------------

    def matthews_corrcoef(self, predicted_classes, labels):
        '''
        Computes the Matthew's correlation coefficient
        from arrays of predicted classes and true labels.
        The predicted_classes may be either raw logits, or
        an array of already processed logits, namely 
        class labels. Ex:
        
           [[logit1,logit2,logit3],
            [logit1,logit2,logit3]
            ]
            
        or: [class1, class2, ...]
        
        where the former are the log odds of each
        class. The latter are the classes decided by
        the highes-logit class. See self.logits_to_classes()
            
        
        @param predicted_classes: array of predicted class labels
        @type predicted_classes: {[int]|[[float]]}
        @param labels: array of true labels
        @type labels: [int]
        @return: Matthew's correlation coefficient,
        @rtype: float
        '''
        if len(predicted_classes.shape) > 1:
            predicted_classes = self.logits_to_classes(predicted_classes)
        mcc = matthews_corrcoef(labels, predicted_classes)
        return mcc
        
        
    #------------------------------------
    # confusion_matrix 
    #-------------------
    
    def confusion_matrix(self, 
                         logits_or_classes, 
                         y_true, 
                         matrix_labels=[0, 1, 2]):
        '''
        Print and return the confusion matrix
        '''
        # Format confusion matrix:
             
        #             right   left    neutral
        #     right
        #     left
        #     neutral

        if len(logits_or_classes.shape) > 1:
            predicted_classes = self.logits_to_classes(logits_or_classes)
         
        n_by_n_conf_matrix = confusion_matrix(y_true, predicted_classes, matrix_labels) 
           
        self.log.info('Confusion Matrix :')
        self.log.info(n_by_n_conf_matrix) 
        self.log.info(f'Accuracy Score :{accuracy_score(y_true, predicted_classes)}')
        self.log.info('Report : ')
        self.log.info(classification_report(y_true, predicted_classes))
        
        return n_by_n_conf_matrix

#     #------------------------------------
#     # compute_matthews_coefficient 
#     #-------------------
#     
#     def compute_matthews_coefficient(self, model, prediction_dataloader):
#         # Tracking variables 
#         predictions , true_labels = [], []
#         
#         # Predict 
#         for batch in prediction_dataloader:
#             # Add batch to GPU
#             batch = tuple(t.to(self.gpu_device) for t in batch)
#             # Unpack the inputs from our dataloader
#             batch['tok_ids'], b_input_mask, b_labels = batch
#             # Telling the model not to compute or store gradients, saving memory and speeding up prediction
#             with torch.no_grad():
#             # Forward pass, calculate logit predictions
#                 logits = model(batch['tok_ids'], token_type_ids=None, attention_mask=b_input_mask)
#             # Move logits and labels to CPU
#             logits = logits[0].detach().cpu().numpy()
#             label_ids = b_labels.to('cpu').numpy()
#           
#             # Store predictions and true labels
#             predictions.append(logits)
#             true_labels.append(label_ids)
#         
#         del batch
#         del logits
#         # From torch:
#         cuda.empty_cache()
#         
#         # Combine the results for all of the batches and calculate our final MCC score.
#         
#         # Combine the results across all batches. 
#         flat_predictions = np.concatenate(predictions, axis=0)
#         
#         # For each sample, pick the label (0 or 1) with the higher score.
#         flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
#         
#         # Combine the correct labels for each batch into a single list.
#         flat_true_labels = np.concatenate(true_labels, axis=0)
#         
#         # Calculate the MCC
#         mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
#         
#         self.log.info('Total MCC: %.3f' % mcc)
#         return mcc

    #------------------------------------
    # print_test_results 
    #-------------------
    
    def print_test_results(self):
        self.log.info(self.training_stats)
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


# ---------------------- Utilities ----------------------

    #------------------------------------
    # prepare_model_save 
    #-------------------
    
    def prepare_model_save(self, model_file):
        if os.path.exists(model_file):
            print(f"File {model_file} exists")
            print("If intent is to load it, go to cell 'Start Here...'")
            self.log.info("Else remove on google drive, or change model_file name")
            self.log.info("Removal instructions: Either execute 'os.remove(model_file)', or do it in Google Drive")
            sys.exit(1)

        # File does not exist. But ensure that all the 
        # directories on the way exist:
        paths = os.path.dirname(model_file)
        try:
            os.makedirs(paths)
        except FileExistsError:
            pass

#     #------------------------------------
#     # to_np_array 
#     #-------------------
# 
#     def to_np_array(self, array_string):
#         # Use the pattern to substitute occurrences of
#         # "123   45" with "123,45". The \1 refers to the
#         # digit that matched (i.e. the capture group):
#         proper_array_str = PoliticalLeaningsAnalyst.SPACE_TO_COMMA_PAT.sub(r'\1,', array_string)
#         # Remove extraneous spaces:
#         proper_array_str = re.sub('\s', '', proper_array_str)
#         # Turn from a string to array:
#         return np.array(ast.literal_eval(proper_array_str))

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
    # accuracy 
    #-------------------

    def accuracy(self, predicted_classes, labels):
        '''
        Function to calculate the accuracy of our predictions vs labels.
        The predicted_classes may be either raw logits, or
        an array of already processed logits, namely 
        class labels. Ex:
        
           [[logit1,logit2,logit3],
            [logit1,logit2,logit3]
            ]
            
        or: [class1, class2, ...]
        
        where the former are the log odds of each
        class. The latter are the classes decided by
        the highes-logit class. See self.logits_to_classes()

        Accuracy is returned as percentage of times the
        prediction agreed with the labels.
        
        @param predicted_classes: raw predictions: for each sample: logit for each class 
        @type predicted_classes:
        @param labels: for each sample: true class
        @type labels: [int]
        '''
        # Convert logits to classe predictions if needed:
        if len(predicted_classes.shape) > 1:
            predicted_classes = self.logits_to_classes(predicted_classes)
        if type(labels) in (torch.Tensor, tf.Tensor):
            labels = labels.numpy()
        # Compute number of times prediction was equal to the label.
        return np.count_nonzero(predicted_classes == labels) / len(labels)

    #------------------------------------
    # logits_to_classes 
    #-------------------
    
    def logits_to_classes(self, logits):
        '''
        Given an array of logit arrays, return
        an array of classes. Example:
        for a three-class classifier, logits might
        be:
           [[0.4, 0.6, -1.4],
            [-4.0,0.7,0.9]
           ]
        The first says: logit of first sample being class 0 is 0.4.
        To come up with a final prediction for each sample,
        pick the highest logit label as the prediction (0,1,...):
          row1: label == 1
          row2: label == 2

        @param logits: array of class logits
        @type logits: [[float]] or tensor([[float]])
        '''
        # Run argmax on every sample's logits array,
        # generating a class in place of each array.
        # The .numpy() turns the resulting tensor to 
        # a numpy array. The detach() is needed to get
        # just the tensors, without the gradient function
        # from the tensor+grad:  
        pred_classes = tf.map_fn(np.argmax, logits.detach(), dtype=np.int16).numpy()
        return pred_classes

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

# # --------------------- LeaningsDataset -----------
# 
# class LeaningsDataset(IterableDataset):
#     '''
#     Takes path to a CSV file prepared by ****????****
#     Columns: id,advertiser,page,leaning,tokens,ids
#     Sample:
#       (10,'Biden','http://...','left',"['[CLS],'Joe','runs',...'[SEP]']",'[[114 321 ...],[4531 ...]])
#     
#     Tokens is a stringified array of array of tokens.
#     Length: sequence size (e.g. 128)
#     Length: as many are there are lines of sample ads.
#     Ids are a stringified arrays of arrays of ints. Each
#       int is an index into BERT vocab. 
#     Length: sequence size (e.g. 128)
#     '''
# 
#     #------------------------------------
#     # Constructor 
#     #-------------------
# 
#     def __init__(self, csv_path):
# 
#         try:
#             csv_fd = open(csv_path, 'r')
#             self.reader = csv.reader(csv_fd)
#         finally:
#             csv_fd.close()
#             
#     #------------------------------------
#     # __iter__ 
#     #-------------------
#     
#     def __iter__(self):
#         return self
# 
#                         
#     #------------------------------------
#     # __next__ 
#     #-------------------
#  
#     def __next__(self):
#         row = next(self.reader)
#         ids = row['ids']
#         
#             
#             
#         reader = pd.read_csv(csv_path,
#                          delimiter=',', 
#                          header=0, 
#                          converters={'ids' : self.to_np_array}
#                         )
#         
#         
#         # Extract the sentences and labels of our training 
#         # set as numpy ndarrays.
#         labels = df.leaning.values
#         # Labels must be int-encoded:
#         label_encodings = []
#         for i in range(len(labels)):
#             if labels[i] == 'right':
#                 label_encodings.append(0)
#             if labels[i] == 'left':
#                 label_encodings.append(1)
#             if labels[i] == 'neutral':
#                 label_encodings.append(2)
#         
#         # Grab the BERT index ints version of the tokens:
#         input_ids = self.train_set.ids
#         
#         # Create attention masks
#         attention_masks = []
#         
#         # Create a mask of 1s for each token followed by 0s for padding
#         for seq in input_ids:
#             #seq_mask = [float(i>0) for i in seq]
#             seq_mask = [int(i>0) for i in seq]
#             attention_masks.append(seq_mask)
#         
#         return (label_encodings, input_ids, attention_masks)
#         
# -------------------- Main ----------------
if __name__ == '__main__':
    
    data_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    csv_path = os.path.join(data_dir, "facebook_ads.csv")
    
    pa = PoliticalLeaningsAnalyst(csv_path,
                                  text_col_name='message',
                                  label_col_name='leaning',
                                  #*********
                                  #epochs=1
                                  #*********
                                  )
    pa.print_test_results()
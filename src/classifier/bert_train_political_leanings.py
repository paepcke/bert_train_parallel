'''
Created on Jun 8, 2020

@author: paepcke
'''

#import ast
from _collections import OrderedDict
import argparse
import datetime
import os, sys
import random
import time

import GPUtil
# Mixed floating point facility (Automatic Mixed Precision)
# From Nvidia: https://nvidia.github.io/apex/amp.html
from apex import amp
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import matthews_corrcoef
from torch import nn, cuda
import torch
from transformers import AdamW, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(__file__))

from bert_feeder_dataloader import BertFeederDataloader
from bert_feeder_dataloader import set_split_id
from bert_feeder_dataset import BertFeederDataset
from logging_service import LoggingService
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ------------------------ Specialty Exceptions --------

class NoGPUAvailable(Exception):
    pass

class TrainError(Exception):
    # Error in the train/validate loop
    pass

# ------------------------ Main Class ----------------
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
    # Modify the following for different, or
    # additional labels (needs testing):
    LABEL_ENCODINGS=OrderedDict({'right'  : 0,
                                 'left'   : 1,
                                 'neutral': 2})

    # Automatic Mixed Precision optimization setting:
    
    #   "00"  Pure 32-bit floating point (F32), therefore a no-op
    #            Use to get lower speed, upper accuracy bound
    #   "01"  Recommended: Mixed precision, amp figures out 
    #           which methods should be run F16, and which F32
    #   "02"  Also a mix; but does not patch pytorch or Tensor funcs
    #   "03"  Pure F16. Fastest, but may not achieve stability.
    #           Try out to get upper speed, lower accuracy bounds
     
    AMP_OPTIMIZATION_LEVEL = "O1"
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 csv_path,
                 model_save_path=None,
                 text_col_name='text',
                 label_col_name='label',
                 epochs=4,
                 batch_size=128,
                 sequence_len=128,
                 learning_rate=3e-5,
                 label_encodings=None,
                 logfile=None,
                 delete_db=False
                 ):
        '''
        Number of epochs: 2, 3, 4 
        '''
        
        if logfile is None:
            default_logfile_name = os.path.join(os.path.dirname(__file__), 'facebook_train.log')
            self.log = LoggingService(logfile=default_logfile_name)
        elif logfile == 'stdout':
            self.log = LoggingService()
        else:
            self.log = LoggingService(logfile=logfile)
        
        self.batch_size = batch_size
        self.epochs     = epochs
        
        if label_encodings is None:
            self.label_encodings = self.LABEL_ENCODINGS
        else:
            self.label_encodings = label_encodings
        # The following call also sets self.gpu_obj
        # to a GPUtil.GPU instance, so we can check
        # on the GPU status along the way:
        
        self.gpu_device = self.enable_GPU()

        if model_save_path is None:
            (csv_file_root, _ext) = os.path.splitext(csv_path)
            model_save_path = csv_file_root + '_trained_model' + '.sav'
        # Preparation:
        dataset = BertFeederDataset(csv_path,
                                    self.label_encodings,
                                    text_col_name=text_col_name,
                                    label_col_name=label_col_name,
                                    sequence_len=sequence_len,
                                    delete_db=delete_db
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
        (predictions, labels) = self.test(model, dataloader)
        
        # EVALUATE RESULT:
        # Calculate the MCC

        mcc = self.matthews_corrcoef(predictions, labels)
        self.log.info(f"Test Matthew's coefficient: {mcc}")
        test_accuracy = self.accuracy(predictions, labels)
        self.log.info(f"Accuracy on test set: {test_accuracy}")
        
        # Save the model:
        
        self.log.info(f"Saving model to {model_save_path} ...")
        with open(model_save_path, 'wb') as fd:
            #torch.save(model, fd)
            torch.save(model.state_dict(), fd)

        # Save the test predictions:
        predictions_path = f"{csv_file_root}_testset_predictions.npy"
        self.log.info(f"Saving predictions to {predictions_path}")
        with open(predictions_path, 'wb') as fd:
            torch.save(predictions, fd)
        
        # Save the training stats:
        training_stats_path = f"{csv_file_root}_train_test_stats.dict"
        with open(training_stats_path, 'wb') as fd:
            torch.save(self.training_stats, fd)
        
#       Note: To maximize the score, we should now merge the 
#       validation set back into the train set, and retrain. 

    #------------------------------------
    # enable_GPU 
    #-------------------

    def enable_GPU(self, raise_gpu_unavailable=True):

        self.gpu_obj = None

        # Get the GPU device name.
        # Could be (GPU available):
        #   device(type='cuda', index=0)
        # or (No GPU available):
        #   device(type='cpu')

        gpu_objs = GPUtil.getGPUs()
        if len(gpu_objs) == 0:
            return 'cpu'

        # GPUs are installed, are any available, given
        # their current memory/cpu usage? We use the 
        # default maxLoad of 0.5 and maxMemory of 0.5
        # as OK to use GPU:
        
        try:
            # If a GPU is available, the following returns
            # a one-element list of GPU ids. Else it throws
            # an error:
            device_id = GPUtil.getFirstAvailable()[0]
        except RuntimeError:
            # If caller wants non-availability of GPU
            # even though GPUs are installed to be an 
            # error, throw one:
            if raise_gpu_unavailable:
                raise NoGPUAvailable("Even though GPUs are installed, all are already in use.")
            else:
                # Else quietly revert to CPU
                return 'cpu'
        
        # Get the GPU object that has the found
        # deviceID:
        for gpu_obj in gpu_objs:
            if gpu_obj.id == device_id:
                self.gpu_obj = gpu_obj
                break
            
        if self.gpu_obj is None:
            # This should not happen:
            raise NoGPUAvailable(f"GPU availability yields dev Id {device_id}; yet none of GPUs has that id")
        
        # Initialize a string to use for moving 
        # tensors between GPU and cpu with their
        # to(device=...) method:
        self.cuda_dev = f"cuda:{device_id}" 
        return device_id 
    
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
            model = model.to(device=self.cuda_dev)
        # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          #lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          lr = learning_rate,
                          eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                        )

        if self.gpu_device != 'cpu':
            # Allow Amp to perform casts as required by the opt_level
            # Second AMP related change:
            (model, optimizer) = amp.initialize(model, 
                                                optimizer, 
                                                opt_level=self.AMP_OPTIMIZATION_LEVEL)
        
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
        - Tell the network to update parameters with optimizer.sample_counter()
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

            # When using a GPU, we check GPU memory 
            # at critical moments, and store the result
            # as a dict in the following list:
            self.gpu_status_history = []

            # For each batch of training data...
            # Tell data loader to pull from the train sample queue:
            dataloader.switch_to_split('train')
            try:
                for sample_counter, batch in enumerate(dataloader):
            
                    # Progress update every 50 batches.
                    if sample_counter % 50 == 0 and not sample_counter == 0:
                        # Calculate elapsed time in minutes.
                        elapsed = self.log.info(f"{self.format_time(time.time() - t0)}") 
                        
                        # Report progress.
                        self.log.info('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(sample_counter, len(dataloader), elapsed))
            
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
                        b_input_ids = batch['tok_ids'].to(device=self.cuda_dev)
                        b_input_mask = batch['attention_mask'].to(device=self.cuda_dev)
                        b_labels = batch['label'].to(device=self.cuda_dev)
            
                    # Always clear any previously calculated gradients before performing a
                    # backward pass. PyTorch doesn't do this automatically because 
                    # accumulating the gradients is "convenient while training RNNs". 
                    # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                    model.zero_grad()        
            
                    # Note GPU usage:
                    if self.gpu_device != 'cpu':
                        self.history_checkpoint(epoch_i, sample_counter,'pre_model_call')
                        
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
                    
                    if self.gpu_device != 'cpu':
                        b_input_ids = b_input_ids.to('cpu')
                        logits = logits.to('cpu')
                        b_labels = b_labels.to('cpu')
                        del b_input_mask
                        cuda.empty_cache()                    
    
                    train_acc = self.accuracy(logits, b_labels)
                    total_train_accuracy += train_acc
    
                    # Note GPU usage:
                    if self.gpu_device != 'cpu':
                        self.history_checkpoint(epoch_i, sample_counter,'post_model_call')
    
                    # Accumulate the training train_loss over all of the batches so that we can
                    # calculate the average train_loss at the end. `train_loss` is a Tensor containing a
                    # single value; the `.item()` function just returns the Python value 
                    # from the tensor.
                    total_train_loss += train_loss.item()
            
                    # Perform a backward pass to calculate the gradients.
                    if self.gpu_device != 'cpu':
                        # Third AMP related change:
                        with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        train_loss.backward()

                    # Clip the norm of the gradients to 1.0.
                    # This is to help prevent the "exploding gradients" problem.
                    # From torch:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
                    if self.gpu_device != 'cpu':
                        del b_input_ids
                        del b_labels
                        del train_loss
                        del logits
                        cuda.empty_cache()
                        self.history_checkpoint(epoch_i, sample_counter,'post_model_freeing')
     
    
                    # Update parameters and take a sample_counter using the computed gradient.
                    # The optimizer dictates the "update rule"--how the parameters are
                    # modified based on their gradients, the learning rate, etc.
                    
                    optimizer.step()
                    
                    # Note GPU usage:
                    if self.gpu_device != 'cpu':
                        cuda.empty_cache()
                        self.history_checkpoint(epoch_i, sample_counter,'post_optimizer')
                        
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
                        b_input_ids = batch['tok_ids'].to(device=self.cuda_dev)
                        b_input_mask = batch['attention_mask'].to(device=self.cuda_dev)
                        b_labels = batch['label'].to(device=self.cuda_dev)
                    
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
                    if self.gpu_device != 'cpu':
                        logits = logits.to('cpu')
                        b_labels = b_labels.to('cpu')
                                            
                    # Accumulate the validation loss and accuracy
                    total_val_loss += val_loss.item()
                    total_val_accuracy += self.accuracy(logits, b_labels)
    
                    if self.gpu_device != 'cpu':
                        del b_input_ids
                        del b_input_mask
                        del b_labels
                        del val_loss
                        del logits
                        cuda.empty_cache()
    
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
            except Exception as e:
                self.log.err(f"GPU memory used at crash time: {self.gpu_obj.memoryUsed}")
                msg = f"During train/validate: {repr(e)}\n"
                if self.gpu_device != 'cpu':
                    msg += "GPU use history:\n"
                    for chckpt_dict in self.gpu_status_history:
                        for event_info in chckpt_dict.keys():
                            msg += f"    {event_info}:      {chckpt_dict[event_info]}\n"
    
                raise TrainError(msg)
                
        self.log.info("")
        self.log.info("Training complete!")
        
        self.log.info("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))
        
        datetime.datetime.now().isoformat()

    #------------------------------------
    # test 
    #-------------------

    def test(self, model, dataloader):
        '''
        Apply our fine-tuned model to generate all_predictions on the test set.
        '''
        
        # Put model in evaluation mode
        model.eval()
        
        # Tracking variables 
        all_predictions , all_labels = [], []
        
        # Predict
        # Batches come as dicts with keys
        # sample_id, tok_ids, label, attention_mask: 
        for batch in dataloader:
            if self.gpu_device == 'cpu':
                b_input_ids = batch['tok_ids']
                b_input_mask = batch['attention_mask']
                b_labels = batch['label']
            else:
                b_input_ids = batch['tok_ids'].to(device=self.cuda_dev)
                b_input_mask = batch['attention_mask'].to(device=self.cuda_dev)
                b_labels = batch['label'].to(device=self.cuda_dev)
            
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
                logits = logits.to('cpu')
                b_labels = b_labels.to('cpu')
                cuda.empty_cache()

            # Get the class prediction from the 
            # logits:
            predictions = self.logits_to_classes(logits)            
            # Store all_predictions and true labels
            all_predictions.extend(predictions)
            labels = b_labels.numpy()
            all_labels.extend(labels)
        
        self.log.info('    DONE applying model to test set.')
        # Ordered list of label ints:
        matrix_labels = list(self.label_encodings.values())
        self.training_stats['Testing'] = \
                {
                    'Test Loss': loss,
                    'Test Accuracy': self.accuracy(all_predictions, all_labels),
                    'Matthews corrcoef': self.matthews_corrcoef(all_predictions, all_labels),
                    'Confusion matrix' : self.confusion_matrix(all_predictions, 
                                                               all_labels, 
                                                               matrix_labels=matrix_labels
                                                               )}

        if self.gpu_device != 'cpu':
            del loss
            cuda.empty_cache()

        return(all_predictions, all_labels)


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
        if type(predicted_classes) == torch.Tensor and \
            len(predicted_classes.shape) > 1:
            predicted_classes = self.logits_to_classes(predicted_classes)
        mcc = matthews_corrcoef(labels, predicted_classes)
        return mcc
        
        
    #------------------------------------
    # confusion_matrix 
    #-------------------
    
    def confusion_matrix(self, 
                         logits_or_classes, 
                         y_true,
                         matrix_labels=None
                         ):

        '''
        Print and return the confusion matrix
        '''
        # Format confusion matrix:
             
        #             right   left    neutral
        #     right
        #     left
        #     neutral

        if matrix_labels is None:
            matrix_labels = self.label_encodings.values()
            
        if type(logits_or_classes) == torch.Tensor and \
            len(logits_or_classes.shape) > 1:
            predicted_classes = self.logits_to_classes(logits_or_classes)
        else:
            predicted_classes = logits_or_classes
         
        n_by_n_conf_matrix = confusion_matrix(y_true, 
                                              predicted_classes, 
                                              labels=matrix_labels
                                              ) 
           
        self.log.info('Confusion Matrix :')
        self.log.info(n_by_n_conf_matrix) 
        self.log.info(f'Accuracy Score :{accuracy_score(y_true, predicted_classes)}')
        self.log.info('Report : ')
        self.log.info(classification_report(y_true, predicted_classes))
        
        return n_by_n_conf_matrix

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
        if type(predicted_classes) == torch.Tensor and \
            len(predicted_classes.shape) > 1:
            # This call will also move the result to CPU:
            predicted_classes = self.logits_to_classes(predicted_classes)
        if type(labels) != np.ndarray:
            if type(labels) == list:
                labels = np.array(labels)
            else:
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
        # Run amax on every sample's logits array,
        # generating a class in place of each array.
        # The detach() is needed to get
        # just the tensors, without the gradient function
        # from the tensor+grad. The [0] is needed, b/c
        # the np.nonzero() returns a tuple whose only
        # element is a one-valued array.
        # There *must* be a more elegant way to do this! 
        pred_classes = [int(np.nonzero(row == np.amax(row))[0]) \
                        for row in logits.detach().numpy()]
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

    #------------------------------------
    # history_checkpoint 
    #-------------------

    def history_checkpoint(self, epoch_num, sample_counter, process_moment):
        # Note GPU usage:
        self.gpu_status_history.append(
            {'epoch_num'      : epoch_num,
             'sample_counter' : sample_counter,
             'process_moment' : process_moment,
             'GPU_free_memory': self.gpu_obj.memoryFree,
             'GPU_memory_used': self.gpu_obj.memoryUsed
             }
            )
# -------------------- Main ----------------
if __name__ == '__main__':

    data_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="BERT-train from CSV, or run saved model."
                                     )

    parser.add_argument('-g', '--logfile',
                        help="path to log file; if 'stdout' direct to display;\n" +\
                             'Default: facebook_ads.log in script file.',
                        default=None);
    parser.add_argument('-t', '--text',
                        help="name of column with text (default 'text')",
                        default='text'
                        )
    parser.add_argument('-l', '--labels',
                        help="name of column with the true labels (default: 'label')",
                        default='label'
                        )
    parser.add_argument('-d', '--deletedb',
                        help="delete current Sqlite db, which contains the CSV content.\n\
                        If deleted, CSV file will be parsed again, else delete/use will \n\
                        be solicited on the command line.",
                        action='store_true',
                        default=False
                        )
    parser.add_argument('csv_path',
                        help='path to csv file to process')

    args = parser.parse_args();

    #**********
    #args.csv_path = os.path.join(data_dir, "facebook_ads_clean.csv")
    #args.text = 'text'
    #args.labels = 'leaning'
    #**********
    
    pa = PoliticalLeaningsAnalyst(args.csv_path,
                                  text_col_name=args.text,
                                  label_col_name=args.labels,
                                  #*********
                                  epochs=1,
                                  #*********
                                  logfile=args.logfile,
                                  delete_db=args.deletedb
                                  )
    pa.print_test_results()
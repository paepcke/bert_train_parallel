# Parallel Bert Training

This pytorch based module starts with a csv file of at least two
columns: text snippets, and labels. The text is tokenized and brought
into a form required by Bert. A Bert model is trained taking the
values of the labels column as truth. A number of performance tests
are performed on the results, whose outputs are written to files. A
result reporting tool can create printed and chart format result
summaries. Summaries include confusion matrix, accuracy, and more.

## Special Features

The module provides a number of features behond the pytorch/apex
facilities.

### Additional Multimachine GPU Usage Controls

GPU use is organized via the pytorch apex DistributedDataParallel
facility, which provides training support on GPUs of a single, or
multiple machines (nodes). However, these modules assume the same
number of GPUs to be installed on each node, and also the same number
to be *used* throughout. This module extends the underlying modules by
providing user control over how many GPUs are used on various
machines.

### Sqlite Database Backing

All information obtained by parsing the input csv file, tokenization,
lookup of token indices into the Bert vocabulary, and splitting into
train/validate/test sets is preserved in an Sqlite database. This
information retention affords significant time savings when training
multiple times with different parameters. Performance test results are
also retained in the database

### Token Folding

Bert training requires the number of tokens per input to be
constant. This `sequence length` constancy is usually achieved by
truncating long text, and padding short text with zeroes. Instead of
truncating, this module folds tokens that exceed the sequence length
into new input rows, which inherit the original text's label. For
example, the following CSV row will be converted into two rows. Assume
sequence length of four:

```
"Beat my Republican opponent! She will remove stop signs from your neighborhood.", left

Turns into

"Beat my Republican opponent!",left
"She will remove stop",left
"signs from your neighborhood",left
```

## Running Example

For a running example suppose we wish to classify Facebook ads by
their political leanings: right, left, neutral. A csv file
`leanings.csv` was created:

```
text,                          label
-------------------------------------
Beat my Republican opponent!  ,left
Democrats want you to...      ,right
Buy this toothpaste!...       ,neutral
           ...
```

The goal is a Bert model that given text, predicts political leaning.

Following workflows 

### Single Machine---As Many GPUs as Available

```
launch.py bert_train_parallel.py leanings.csv
```

The resulting files will be deposited into the directory where the
`leanings.csv` file resides.

- `leanings_testset_predictions.csv` will contain test set predictions
  and truth labels.

- `leanings_train_test_stats.json` will contain training and
  validation losses and accuracy for each epoch.

- `leanings_trained_model.sav` will hold the trained model, readable
  via torch.load().

- `leanings.sqlite` will allow another run to bypass csv processing.

### Single Machine---Use Only 2 GPUs


```
launch.py --gpus_here 2 bert_train_parallel.py leanings.sqlite
```

The least busy GPUs will be selected. Note the use of
`leanings.sqlite` instead of `leanings.csv`. This example assumes that
the Sqlite database was created earlier. If this is not the case, use
the .csv extension.

### Multiple Machines---As Many GPUs as Available

The following assumes that node 0 contains 3 GPUs, while node 1
containes 1 GPU.

Node 0:
```
launch.py --gpus_other 1 bert_train_parallel.py leanings.sqlite
```
Node 1:
```
launch.py --node_rank 1 \
          --master_addr my.machine.com \
          --other_gpus 3 \
          bert_train_parallel.py leanings.sqlite
```

While the node rank being 0 is default in the upper command, on node 1
the rank must be provided. Similarly, each node other than the master
node need to be informed of the master's IP address or host name.

Finally, each machine must be told how many GPUs will be in use across
all the machines, other than itself.

### Multiple Machines---Fixed Number of GPUs on Each

If only, say 1 GPU is to be used on node 0, even though three are
available, the following will accomplish the goal:

Node 0:
```
launch.py --gpus_other 1 --gpus_here 1 bert_train_parallel.py leanings.sqlite
```
Node 1:
```
launch.py --node_rank 1 \
          --master_addr my.machine.com \
          --other_gpus 1 \
          bert_train_parallel.py leanings.sqlite
```

### Configuration File for GPU Use

Instead of specifying on the command line of each machine how many
GPUs are used locally and elsewhere, a JSON configuration file can
contain that information. For example:

```
   {"foo.bar.com" : 4,
    "127.0.0.1"   : 5,
    "172.12.145.1" : 6
   }
```

where the integers are the number of GPUs to use on each
machine. Localhost may be specified as 'localhost', '127.0.0.1', or it
may be left out. In the latter case, all the GPUs at the local node
are assumed to be usable.

## Checking Results

The `bert_train_parallel.py` generates several files, as explained
above. The module `bert_result_analysis.py` consumes those files, and
produces a number of common measures and diagnostics:




## Distribution Model and Terminology

Different examples across the Web use slightly different modules, and
thereby terminology. There is a summary of how this module operates. 

## Terminology

- *Node* is synonymous with one computer. Each node may own none, one,
  or multiple GPUs.

- *Process* every GPU is used by exactly one process in the Unix sense
   of the word.

- *Sample* corresponds to one line in the original CSV file. Though in
   reality the words in the file are replaced by integer indices into
   the Bert vocabulary.

- *Batch size* is the number of samples that are provided to the
   evolving model at once. Typical numbers are 1, 4, 16, 32, 64. The
   higher the number, the faster the training, but the more memory is
   required in the GPU units. Batches between [2,N-1] with N being the
   total number of samples are called *mini-batches*. Each training
   step uses all of the samples in a (mini-) batch, combining the loss
   encountered by each of the batch member samples to a single value.

- *Epoch* is the process of running through all the samples, and
   presenting them to the model for training. The training process may
   run through many epochs. For each epoch the order in which samples
   are prosented to the model in training is changed. The number of
   necessary epochs depends on whether a model is being trained from
   scratch, or only a bit of training is needed on top of a pretrained
   model. For Bert, 1 to 4 epochs seems typical.

- *Distributed sampling* ensures that each GPU only works on a
   predictable subset of samples, no matter on which machine the GPU
   resides. The result is that all samples are used to train the
   model, and that each sample is only presented to the evolving model
   once within one epoch.

   This coordination is possible because:

    1. Each node knows how many GPUs exist in the 'world' of the
       training execution. This knowledge allows each process to
       determine how many samples to work on during training: N/G,
       where N is the number of samples in the training set, and G is
       the total number of involved GPUs.

    2. The nodes share random sampling seeds, which allows them to
       ensure that no sample is used twice. The epoch number is used
       as the seed. The number therefore does not need to be
       communicated.

- *Master node*: one node is special. It coordinates the only
   communication needed among nodes: the model parameters, which need
   to be synced. This is why all other nodes must be informed of the
   master node: IP or name.
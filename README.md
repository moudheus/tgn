# TGN: Temporal Graph Networks (modified)


This repo is a modified fork of the TGN repo supporting the Temporal Graph Networks paper by Rossi and al:

- Reference repo: https://github.com/twitter-research/tgn
- Reference paper: https://arxiv.org/abs/2006.10637


The following modifications were made to support comparison with simpler temporal graph methods for link prediction:

- A new script for global future predictions (for each user in the training set, outputs the top 100 recommended items for a given global test time in the future) : [predict.py](https://github.com/moudheus/tgn/blob/master/predict.py)
- A new function to support creation of embeddings for global prediction: [model/tgn.py](https://github.com/moudheus/tgn/blob/master/model/tgn.py#L272)
- Disabling of training data modification for new nodes as we focus on transductive link prediction: [util/data_processing.py](https://github.com/moudheus/tgn/blob/master/utils/data_processing.py#L97)


## Running the experiments

### Requirements

Dependencies (with python >= 3.7):

```{bash}
pandas==1.1.0
torch==1.6.0
scikit_learn==0.23.1
```

### Dataset and Preprocessing

#### Download the public data

Download the sample datasets (eg. wikipedia and reddit).

```
source ./download_data.sh 
```

#### Run all

To run the main models on all datasets with future prediction use the script below.

```
source ./run.sh 
```

If you want to run specific parts see instructions below.

#### Preprocess the data

We use the dense `npy` format to save the features in binary format. If edge features or nodes 
features are absent, they will be replaced by a vector of zeros. 
```{bash}
python utils/preprocess_data.py --data wikipedia --bipartite
python utils/preprocess_data.py --data reddit --bipartite
```

### Model Training

Self-supervised learning using the link prediction task:

```{bash}
python train_self_supervised.py --data wikipedia --use_memory --prefix tgn-attn --n_runs 10
```

### Prediction (new)

Predict top 100 items for each user:

```{bash}
python predict.py --data wikipedia --use_memory --prefix tgn-attn 
```

### Baselines

```{bash}

# Jodie
python train_self_supervised.py --data wikipedia --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10

# DyRep
python train_self_supervised.py --data wikipedia --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --n_runs 10
```

### Ablation Study

Commands to replicate all results in the ablation study over different modules:
```{bash}
# TGN-2l
python train_self_supervised.py --data wikipedia --use_memory --n_layer 2 --prefix tgn-2l --n_runs 10 

# TGN-no-mem
python train_self_supervised.py --data wikipedia --prefix tgn-no-mem --n_runs 10 

# TGN-time
python train_self_supervised.py --data wikipedia --use_memory --embedding_module time --prefix tgn-time --n_runs 10 

# TGN-id
python train_self_supervised.py --data wikipedia --use_memory --embedding_module identity --prefix tgn-id --n_runs 10

# TGN-sum
python train_self_supervised.py --data wikipedia --use_memory --embedding_module graph_sum --prefix tgn-sum --n_runs 10

# TGN-mean
python train_self_supervised.py --data wikipedia --use_memory --aggregator mean --prefix tgn-mean --n_runs 10
```


#### General flags

```{txt}
optional arguments:
  -d DATA, --data DATA         Data sources to use (wikipedia or reddit)
  --bs BS                      Batch size
  --prefix PREFIX              Prefix to name checkpoints and results
  --n_degree N_DEGREE          Number of neighbors to sample at each layer
  --n_head N_HEAD              Number of heads used in the attention layer
  --n_epoch N_EPOCH            Number of epochs
  --n_layer N_LAYER            Number of graph attention layers
  --lr LR                      Learning rate
  --patience                   Patience of the early stopping strategy
  --n_runs                     Number of runs (compute mean and std of results)
  --drop_out DROP_OUT          Dropout probability
  --gpu GPU                    Idx for the gpu to use
  --node_dim NODE_DIM          Dimensions of the node embedding
  --time_dim TIME_DIM          Dimensions of the time embedding
  --use_memory                 Whether to use a memory for the nodes
  --embedding_module           Type of the embedding module
  --message_function           Type of the message function
  --memory_updater             Type of the memory updater
  --aggregator                 Type of the message aggregator
  --memory_update_at_the_end   Whether to update the memory at the end or at the start of the batch
  --message_dim                Dimension of the messages
  --memory_dim                 Dimension of the memory
  --backprop_every             Number of batches to process before performing backpropagation
  --different_new_nodes        Whether to use different unseen nodes for validation and testing
  --uniform                    Whether to sample the temporal neighbors uniformly (or instead take the most recent ones)
  --randomize_features         Whether to randomize node features
  --dyrep                      Whether to run the model as DyRep
```

## TODOs 

* Make code memory efficient: for the sake of simplicity, the memory module of the TGN model is 
implemented as a parameter (so that it is stored and loaded together of the model). However, this 
does not need to be the case, and 
more efficient implementations which treat the models as just tensors (in the same way as the 
input features) would be more amenable to large graphs.

## Cite us

```bibtex
@inproceedings{tgn_icml_grl2020,
    title={Temporal Graph Networks for Deep Learning on Dynamic Graphs},
    author={Emanuele Rossi and Ben Chamberlain and Fabrizio Frasca and Davide Eynard and Federico 
    Monti and Michael Bronstein},
    booktitle={ICML 2020 Workshop on Graph Representation Learning},
    year={2020}
}
```



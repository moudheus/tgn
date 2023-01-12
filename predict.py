#!/usr/bin/env python
# coding: utf-8

import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from tqdm import trange
import time

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics


def rank_score(preds_list, true_set):
    for i, pred in enumerate(preds_list):
        if pred in true_set:
            return i+1
    return 1_000_000


def write_dict(d, path):
    with open(path, 'w') as f:
        json.dump(d, f)
        f.write('\n')


torch.manual_seed(0)
np.random.seed(0)



### Argument and global variables
parser = argparse.ArgumentParser("TGN self-supervised training")
parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="Dataset name (eg. wikipedia or reddit)",
    default="wikipedia",
)
parser.add_argument("--q1", type=float, default=0.70, help="Quantile for end of train")
parser.add_argument("--q2", type=float, default=0.85, help="Quantile for end of val")

parser.add_argument("--n_pred_windows", type=int, default=1, help="number of prediction windows inside the test window")

parser.add_argument("--bs", type=int, default=200, help="Batch_size")
parser.add_argument(
    "--prefix", type=str, default="", help="Prefix to name the checkpoints"
)
parser.add_argument(
    "--n_degree", type=int, default=10, help="Number of neighbors to sample"
)
parser.add_argument(
    "--n_head", type=int, default=2, help="Number of heads used in attention layer"
)
parser.add_argument("--n_layer", type=int, default=1, help="Number of network layers")
parser.add_argument("--drop_out", type=float, default=0.1, help="Dropout probability")
parser.add_argument("--gpu", type=int, default=0, help="Idx for the gpu to use")
parser.add_argument(
    "--node_dim", type=int, default=100, help="Dimensions of the node embedding"
)
parser.add_argument(
    "--time_dim", type=int, default=100, help="Dimensions of the time embedding"
)
parser.add_argument(
    "--use_memory",
    action="store_true",
    help="Whether to augment the model with a node memory",
)
parser.add_argument(
    "--embedding_module",
    type=str,
    default="graph_attention",
    choices=["graph_attention", "graph_sum", "identity", "time"],
    help="Type of embedding module",
)
parser.add_argument(
    "--message_function",
    type=str,
    default="identity",
    choices=["mlp", "identity"],
    help="Type of message function",
)
parser.add_argument(
    "--memory_updater",
    type=str,
    default="gru",
    choices=["gru", "rnn"],
    help="Type of memory updater",
)
parser.add_argument(
    "--aggregator", type=str, default="last", help="Type of message " "aggregator"
)
parser.add_argument(
    "--memory_update_at_end",
    action="store_true",
    help="Whether to update memory at the end or at the start of the batch",
)
parser.add_argument(
    "--message_dim", type=int, default=100, help="Dimensions of the messages"
)
parser.add_argument(
    "--memory_dim",
    type=int,
    default=172,
    help="Dimensions of the memory for " "each user",
)
parser.add_argument(
    "--different_new_nodes",
    action="store_true",
    help="Whether to use disjoint set of new nodes for train and val",
)
parser.add_argument(
    "--uniform",
    action="store_true",
    help="take uniform sampling from temporal neighbors",
)
parser.add_argument(
    "--randomize_features",
    action="store_true",
    help="Whether to randomize node features",
)
parser.add_argument(
    "--use_destination_embedding_in_message",
    action="store_true",
    help="Whether to use the embedding of the destination node as part of the message",
)
parser.add_argument(
    "--use_source_embedding_in_message",
    action="store_true",
    help="Whether to use the embedding of the source node as part of the message",
)
parser.add_argument(
    "--dyrep", action="store_true", help="Whether to run the dyrep model"
)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)



BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim


Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)

FULL_PREFIX = f"{args.prefix}-{args.data}"
MODEL_SAVE_PATH = f"./saved_models/{FULL_PREFIX}.pth"
#RESULTS_PATH = Path(f'results-timing-w{args.n_pred_windows}')
RESULTS_PATH = Path('results')



### set up logger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler("log/{}.log".format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)



### Extract data for training, validation and testing
(
    node_features,
    edge_features,
    full_data,
    train_data,
    val_data,
    test_data,
    new_node_val_data,
    new_node_test_data,
) = get_data(
    DATA,
    different_new_nodes_between_val_and_test=args.different_new_nodes,
    randomize_features=args.randomize_features,
    q1=args.q1,
    q2=args.q2,
)


# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)


# Set device
device_string = "cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu"
device = torch.device(device_string)

# Compute time statistics
(
    mean_time_shift_src,
    std_time_shift_src,
    mean_time_shift_dst,
    std_time_shift_dst,
) = compute_time_statistics(
    full_data.sources, full_data.destinations, full_data.timestamps
)


RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# Initialize Model
tgn = TGN(
    neighbor_finder=train_ngh_finder,
    node_features=node_features,
    edge_features=edge_features,
    device=device,
    n_layers=NUM_LAYER,
    n_heads=NUM_HEADS,
    dropout=DROP_OUT,
    use_memory=USE_MEMORY,
    message_dimension=MESSAGE_DIM,
    memory_dimension=MEMORY_DIM,
    memory_update_at_start=not args.memory_update_at_end,
    embedding_module_type=args.embedding_module,
    message_function=args.message_function,
    aggregator_type=args.aggregator,
    memory_updater_type=args.memory_updater,
    n_neighbors=NUM_NEIGHBORS,
    mean_time_shift_src=mean_time_shift_src,
    std_time_shift_src=std_time_shift_src,
    mean_time_shift_dst=mean_time_shift_dst,
    std_time_shift_dst=std_time_shift_dst,
    use_destination_embedding_in_message=args.use_destination_embedding_in_message,
    use_source_embedding_in_message=args.use_source_embedding_in_message,
    dyrep=args.dyrep,
)
tgn = tgn.to(device)

num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

logger.info("num of training instances: {}".format(num_instance))
logger.info("num of batches per epoch: {}".format(num_batch))
idx_list = np.arange(num_instance)


logger.info(f"Loading model {MODEL_SAVE_PATH}")
tgn.load_state_dict(torch.load(MODEL_SAVE_PATH))
logger.info("Done")
tgn.eval()

tgn.embedding_module.neighbor_finder = full_ngh_finder


TOPK = 100


def predict(tgn, sources, destinations, edge_time, n_neighbors):
    source_embeddings = tgn.compute_temporal_embeddings_for_prediction(
        sources,
        np.repeat(edge_time, len(sources)),
        n_neighbors
    )
    destination_embeddings = tgn.compute_temporal_embeddings_for_prediction(
        destinations,
        np.repeat(edge_time, len(destinations)),
        n_neighbors
    )
    scores = []
    for i in trange(len(sources)):
        e = source_embeddings[i, :].repeat(len(destination_embeddings), 1)
        score = tgn.affinity_score(e, destination_embeddings).squeeze(dim=0)
        df = pd.DataFrame()
        df['source'] = np.repeat(sources[i], len(destinations))
        df['destination'] = destinations
        df['score'] = score.detach().cpu().numpy()
        topk = df.sort_values('score', ascending=False).head(TOPK)
        scores.append(topk)

    scores = pd.concat(scores).reset_index(drop=True)
    return scores


def predict_low_mem(tgn, sources, destinations, edge_time, n_neighbors):
    # TODO: compute on the fly to reduce memory usage
    pass


sources = np.unique(train_data.sources)
destinations = np.unique(train_data.destinations)


N_WINDOWS = args.n_pred_windows
ts = test_data.timestamps
edge_times = np.linspace(ts.min(), ts.max(), N_WINDOWS + 1)

for i in range(N_WINDOWS):
    
    t0, t1 = edge_times[i], edge_times[i+1]
    edge_time = (t0 + t1) / 2
    i_window = (t0 <= ts) & (ts <= t1)
    
    t0 = time.time()
    
    scores = predict(tgn, sources, destinations, edge_time, NUM_NEIGHBORS)

    t1 = time.time()

    path = RESULTS_PATH / f'{FULL_PREFIX}-predictions-{i}.csv'
    logger.info(f'Saving to {path}')
    scores.to_csv(path, index=False)

    print(f'Inference took {t1-t0:.1f}s')

    true = pd.DataFrame()
    true['source'] = test_data.sources[i_window]
    true['destination'] = test_data.destinations[i_window]
    true_path = RESULTS_PATH / f'{FULL_PREFIX}-true-{i}.csv'
    logger.info(f'Saving to {true_path}')
    true.to_csv(true_path, index=False)

    # Scoring

    #TODO option to remove training interactions from reco_dict so we only recommand new items
    reco_dict = scores.groupby('source')['destination'].apply(list).to_dict()
    true_dict = true.groupby('source')['destination'].apply(set).to_dict()

    support = set(reco_dict.keys()) & set(true_dict.keys())
    print(f'Support: {len(support)} users')

    rank_dict = {user: rank_score(reco_dict[user], true_dict[user]) for user in support}

    d = args.__dict__.copy()
    d['inference_time'] = t1 - t0

    d['mrr'] = np.mean([1/r for r in rank_dict.values()])
    for k in [1, 3, 10, 25, 50, TOPK]:
        d[f'hit{k}'] = np.mean([r <= k for r in rank_dict.values()])

    for k, v in d.items():
        print(f'{k:<20} : {v:<16}')

    path = RESULTS_PATH / f'{FULL_PREFIX}-metrics-{i}.json'
    write_dict(d, path)
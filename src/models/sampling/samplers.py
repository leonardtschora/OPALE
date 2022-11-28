import scipy.stats as stats, numpy as np, pandas

from src.models.sampling.structure_sampler import structure_sampler, double_structure_sampler
from src.models.sampling.CNN_layers import CNN_structure_sampler
from src.models.sampling.combined_sampler import combined_sampler
from src.models.sampling.regularization_sampler import regularization_sampler
from src.models.sampling.discrete_log_uniform import discrete_loguniform    

def DNN_space(n, country, fast=False, stop_after=-1):
    space = {
        "neurons_per_layer" :  combined_sampler(
            [structure_sampler(1),
             structure_sampler(2),
             structure_sampler(3),],
            weights = [4, 2, 1]),
        "default_activity_regularizer" : combined_sampler(
            [None,
             regularization_sampler(types="all", alpha_scale="log"),
             regularization_sampler(types="l1", alpha_scale="log"),
             regularization_sampler(types="l2", alpha_scale="log")],
            weights = [3, 1, 1, 1]),
        "dropout_rate" : combined_sampler(
            [stats.uniform(0, 0.5), 0], weights=[2, 1]),
        "batch_norm" : stats.bernoulli(0.5),
        "batch_size" : discrete_loguniform(10, n+1),
        "scaler" : ["BCM", "Standard", "Median", "SinMedian"],
        "transformer" : ["BCM", "Standard", "Median", "SinMedian"],
        "stop_after" : [stop_after]}
    if stop_after > 0: space["stop_threshold"] = [1]
    if fast: space["n_epochs"] = [2]
    if fast: space["early_stopping"] = [""]
    return space

def CNN_space(n, W, H, country, fast=False, stop_after=-1):
    space = {
        "structure" : combined_sampler(
            [CNN_structure_sampler(
                1, W, H, 1, mm_filters=(3, 25), max_kernel=(int(W/10),int(H/10)),
                mm_layers=(1, 1),
                min_strides=(4, 2), max_strides=(16, 8),
                min_pool_sizes=(4, 2), max_pool_sizes=(12, 6),
                max_dilation=(int(W/4), int(H/4))),
             CNN_structure_sampler(
                 2, W, H, 1,mm_filters=(3, 25), max_kernel=(int(W/10),int(H/10)),
                 mm_layers=(1, 1),
                min_strides=(4, 2), max_strides=(16, 8),
                 min_pool_sizes=(4, 2), max_pool_sizes=(12, 6),
                 max_dilation=(int(W/4), int(H/4))),
             CNN_structure_sampler(
                 3, W, H, 1,mm_filters=(3, 25), max_kernel=(int(W/10),int(H/10)),
                 mm_layers=(1, 1),
                 min_strides=(4, 2), max_strides=(16, 8),
                 min_pool_sizes=(4, 2), max_pool_sizes=(12, 6),
                 max_dilation=(int(W/4), int(H/4))),
             CNN_structure_sampler(
                 4, W, H, 1,mm_filters=(3, 25), max_kernel=(int(W/10),int(H/10)),
                 mm_layers=(1, 1),
                 min_strides=(4, 2), max_strides=(16, 8),
                 min_pool_sizes=(4, 2), max_pool_sizes=(12, 6),
                 max_dilation=(int(W/4), int(H/4)))             
            ],
            weights = [1, 2, 2, 2]),
        "neurons_per_layer" : combined_sampler(
            [structure_sampler(0),
             structure_sampler(1)],
            weights=[4, 1]),
        "default_activity_regularizer" : combined_sampler(
                [None,
                 regularization_sampler(types="all", alpha_scale="log"),
                 regularization_sampler(types="l1", alpha_scale="log"),
                 regularization_sampler(types="l2", alpha_scale="log")],
                weights = [3, 1, 1, 1]),
        "dropout_rate" : combined_sampler(
            [stats.uniform(0, 0.5), 0], weights=[2, 1]),
        "batch_norm" : stats.bernoulli(0.5),
        "batch_size" : discrete_loguniform(10, n+1),
        "scaler" : ["BCM", "Standard", "Median", "SinMedian"],
        "transformer" : ["BCM", "Standard", "Median", "SinMedian"],
        "stop_after" : [stop_after]}
    if stop_after > 0: space["stop_threshold"] = [1]
    if fast: space["n_epochs"] = [2]
    if fast: space["early_stopping"] = [""]    
    return space

def GNN_space(n, country, fast=False, stop_after=-1, node_level=False):
    space= {
        "channels" :  combined_sampler(
            [structure_sampler(4, min_=24, max_=96),
             structure_sampler(2, min_=24, max_=96),
             structure_sampler(1, min_=24, max_=96),             
             structure_sampler(3, min_=24, max_=96)]),
        "edge_channels" : [24],        
        #"layer_type" : ["ENEEConv", "NEConv"],
        #"layer_type" : ["GATv2Conv", "NNConv", "GINConv", "GCNConv", "NEConv"],
        "regr" : combined_sampler(
            [structure_sampler(2, min_=24, max_=96),
             structure_sampler(1, min_=24, max_=96),
             ()], weights=[1, 1, 1]),
        "batch_norm" : stats.bernoulli(0.5),
        "activation" : ["ReLU", "LeakyReLU", "PReLU"],
        "dropout_rate" : combined_sampler(
            [stats.uniform(0, 0.5), 0], weights=[1, 2]),
        "batch_size" : discrete_loguniform(10, n+1),
        "scaler" : ["BCM", "Standard", "Median", "SinMedian"],
        "transformer" : ["BCM", "Standard", "Median", "SinMedian"],
        "heads" : stats.randint(1, 20),
    }        
    if fast: space["n_epochs"] = [2]
    if fast: space["early_stopping"] = [""]    
    return space

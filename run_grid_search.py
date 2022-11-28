import os
os.environ["OPALE"] = os.curdir
from src.models.nn_wrapper import DNNWrapper, CNNWrapper
from src.models.gnn_wrapper import GNNWrapper
from src.models.model_utils import run

"""
This script performs the hyper-parameter grid search for the DNN, CNN and GNN.
The sampling strategies and search spaces are defined in models/sampling/samplers.
This will overwrite files in data/Grid Search, copy them or set restart = FALSE.

We obtained our results using 24h per model on a 20cpu machine. To run it faster, lower the number of hyperparameter combinations to try 'n_combi' or direclty re-use our
grid search results.

Lower the number of cpus in case of high memory usage.
"""

kwargs = {
    # TASKS
    "GRID_SEARCH" : True,

    # GENERAL PARAMS
    "n_val" : 365,
    "models" : (
        [DNNWrapper, {
            "n_cpus" : 5,
            "n_combis" : 4027,
            "replace_ATC" : ""}],
        [CNNWrapper, {
            "n_cpus" : 5,
            "n_combis" : 503,
            "replace_ATC" : ""}],        
        [GNNWrapper, {
            "n_cpus" : 1, # Internally parallelized
            "n_combis" : 467,            
            "replace_ATC" : ""}],
    ),
    
    # GRID SEARCH PARAMS
    "restart" : True,
}
run(**kwargs)

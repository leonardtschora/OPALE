import os
os.environ["OPALE"] = os.curdir
from src.models.nn_wrapper import DNNWrapper, CNNWrapper
from src.models.gnn_wrapper import GNNWrapper
from src.models.model_utils import run

"""
This script produces the forecasts on the test set using recalibration. 
"""

kwargs = {
    # TASKS
    "RECALIBRATE" : True,

    # GENERAL PARAMS
    "name" : "TSCHORA",
    "datasets" : ("AllEuropeGNN",),
    "base_dataset_names" : ("AllEuropeGNN",),
    "n_val" : 365,
    "models" : (
        [DNNWrapper, {"n_cpus" : 5, "replace_ATC" : ""}],
        [CNNWrapper, {"n_cpus" : 5, "replace_ATC" : ""}],        
        [GNNWrapper, {"n_cpus" : 1, "replace_ATC" : ""}],
    ),
    
    # RECALIBRATION PARAMS      
    "start" : 0,
    "step" : 30,
    "stop" : 30,        
    "calibration_window" : 1440,    
}
run(**kwargs)

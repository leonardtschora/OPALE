import os
os.environ["OPALE"] = os.curdir
from src.models.nn_wrapper import DNNWrapper, CNNWrapper
from src.models.gnn_wrapper import GNNWrapper
from src.models.model_utils import run

"""
This script produces the forecasts on the test set using recalibration. It will read the best hyper-parameter configuration from the files in 'Grid Search' and then it will train the model using it. ATC in the datasets are replaced by the specified 'flow_estimation' ("" stands for using ATCs). 

This scripts executed in 88h on a 20-cpu machine. Delete some models or increase the 'step' to make it faster.
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
        [DNNWrapper, {"n_cpus" : 5, "flow_estimation" : ""}],
        [DNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Flin"}],
        [DNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Flsq"}],
        [DNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Fcmb"}],
        [DNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Funi"}],

        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : ""}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Flin"}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Flsq"}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Fcmb"}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Funi"}],

        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : ""}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Flin"}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Flsq"}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Fcmb"}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Funi"}],
    ),
    
    # RECALIBRATION PARAMS      
    "start" : 0,
    "step" : 30,
    "stop" : 731,        
    "calibration_window" : 1440,    
}
run(**kwargs)

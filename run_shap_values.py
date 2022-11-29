import os
os.environ["OPALE"] = os.curdir
from src.models.nn_wrapper import DNNWrapper, CNNWrapper
from src.models.gnn_wrapper import GNNWrapper
from src.models.model_utils import run

"""
This script produces the shap values for the 30 first days of the test dataset.

This scripts executed in 43h on a 20-cpu machine. Delete some models or lower the 'n_shap' parameter to lower it.
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
        [DNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Fos"}],

        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : ""}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Flin"}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Flsq"}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Fcmb"}],
        [CNNWrapper, {"n_cpus" : 5, "flow_estimation" : "Fos"}],

        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : ""}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Flin"}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Flsq"}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Fcmb"}],
        [GNNWrapper, {"n_cpus" : 1, "flow_estimation" : "Fos"}],
    ),
    
    # RECALIBRATION PARAMS
    "n_shap" : 500,
    "start" : 0,
    "step" : 30,
    "stop" : 30,    
    "calibration_window" : 1440,    
}
run(**kwargs)

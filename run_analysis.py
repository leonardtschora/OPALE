import os
os.environ["OPALE"] = os.curdir

"""
This script contains metric computation, DM tests and shap values analysis.
Make sure that the recalibrated predictions and shap values are generated prior using, or paste them from our data.
"""

import pandas, numpy as np, os, time, math

from src.models.nn_wrapper import DNNWrapper, CNNWrapper
from src.models.gnn_wrapper import GNNWrapper
from src.models.model_utils import run

from src.analysis.metrics_utils import *
from src.analysis.shap_utils import *

################## Global analysis (10s)
# Load real prices
Y = load_real_prices()
nd, ny = Y.shape

# Load recalibrated predictions
models = {"" : [DNNWrapper, CNNWrapper, GNNWrapper],
          "Flin" : [DNNWrapper, CNNWrapper, GNNWrapper],
          "Flsq" : [DNNWrapper, CNNWrapper, GNNWrapper],
          "Fcmb" : [DNNWrapper, CNNWrapper, GNNWrapper],
          "Fos" : [DNNWrapper, CNNWrapper, GNNWrapper]}
predictions, model_wrappers, nv, nm = load_forecasts(models, nd, ny)

# Compute metrics
results = compute_metrics(Y, predictions, nm, model_wrappers)

# Compute DM tests
dm = compute_DM_tests(Y, predictions, nm, nv, model_wrappers, models)
################## Analyse zone by zone (10s)
zones = get_zones(model_wrappers)

# Metrics
results = metric_by_zone(Y, zones, predictions, CC, model_wrappers)
plot_metric_by_zones(results, CC)

# DM tests
pvalues = compute_DM_tests_by_zones(Y, predictions, len(zones), model_wrappers) 
plot_DM_results_by_zone(model_wrappers, zones, pvalues)
################## SHAP Values (3 min)
start = time.time()
all_shaps = load_shaps(models)

# Feature by feature
results = contribution_by_features(models, all_shaps)
plot_features_summary(results)

# Contribution difference between flow estimates and ATC
df = compute_differences(all_shaps, models)
plot_shap_move_by_zone(df)
stop = time.time()

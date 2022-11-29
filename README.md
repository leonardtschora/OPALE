# OPALE
##Optimize &amp; Predict Applied to Leverage European electricity market constraints.

This repository contains the code used for the experiment and analysis of our paper : Forecasting Electricity Prices: an Optimize then Predict-based approach.

## Prerequistes

1. Clone this project.
2. Install dependancies listed in `setup.py` (python > 3).
3. Download data from the [archive](https://www.dropbox.com/sh/c6ea7shsulwgebm/AACb1VxPpkN0ZGY-bpSQ4KhGa?dl=0). Place the **data** folder at the root of it.

## Scripts

5 scripts are available :

- `run_optimize_flows.py` : Computes `Flin`, `Flsq`, `Fcmb`, `Fos`
- `run_grid_search.py` : Evaluates hyper-parameters combinations. Will produce `results.csv`files in the **Grid Search** folder.
- `run_recalibration.py` : Forecasts prices using recalibration. Will produce `predictions.csv` files in the **Predictions** folder.
- `run_shap_values.py` : Computes shap values for the first 30 days of the test set. Will produce `shap_values.npy` files in the **Shap Values** folder.
- `run_analysis.py` : Computes metrics, performs DM tests and plots shap values.

We recommend to execute those scripts in the terminal using the python or ipython interpreters, as results are not displayed.

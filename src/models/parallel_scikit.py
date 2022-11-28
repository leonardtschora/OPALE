import time, pandas, copy, sklearn, warnings, shap, numpy as np, sys, torch
from joblib import Parallel, delayed
from warnings import simplefilter

from sklearn.model_selection import ParameterSampler
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from src.analysis.evaluate import CC

## Seed settings
import tensorflow as tf
import os
os.environ['PYTHONHASHSEED'] = '90125' 
import numpy
from numpy.random import seed
import random

def set_all_seeds(SEED=90125, env=True):
    seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)    
    if env: os.environ['HYPEROPT_FMIN_SEED'] = f"{SEED}"      

def get_param_list_and_seeds(distributions, n_combis, model_wrapper=None,
                             restart=True):
    # Udpate the number of combinations to sample if warm start
    if not restart:
        previous = model_wrapper.load_results()
        n_tested = previous.shape[0]
        print(f"ALREADY TESTED {n_tested} COMBIS")
    else:
        n_tested = 0
        
    # Handle a search space
    param_list = list(ParameterSampler(distributions,
                                       n_iter=n_combis + n_tested))
    seeds = get_seeds(n_combis + n_tested)

    # Only select the last n_combis combinations (in case of warm restart)
    param_list = param_list[-n_combis:]
    seeds = seeds[-n_combis:]
    
    # Get a default value for the TimeStopping criterion
    if "stop_after" in distributions.keys():
        if distributions["stop_after"][0] > 0:
            for p in param_list:
                p["stop_threshold"] = 100

    # Flatten the param list (ex : structure in the CNN)
    flatten_param_list(param_list)        
    return param_list, seeds

def flatten_param_list(param_list):
    for i, psample in enumerate(param_list):
        keys_ = [k for k in psample.keys()]
        for k in keys_:
            if type(psample[k]) is dict:
                for k2 in psample[k].keys():
                    param_list[i][k2] = psample[k][k2]
                del param_list[i][k]
    

def get_seeds(n_combis):
    return [90125 + i for i in range(n_combis)]

def inner_validation(validation_mode, X, y, Xv, yv, model):
    # Check arguments
    if (validation_mode == "external") and ((Xv is None) or (yv is None)):
        raise(Exception("Validation externe et Xv yv non specifies!"))
    
    if ((validation_mode == "oob") or (validation_mode == "internal")):
        if model.spliter is None:
            raise(Exception("Validation interne oob et model spliter non-specifie"))
        if (Xv is not None) or (yv is not None):
            print("Validation interne oob, Xv et yv sont ignores")

    # Compute the same validation set that will be used internaly
    if ((validation_mode == "oob") or (validation_mode == "internal")):
        ((Xt, yt), (Xv, yv)) = model.spliter(X, y)

    # Does nothing is validation mode is external
    return Xv, yv

def outer_validation(validation_mode, external_spliter, X, y):
    if (validation_mode == "external") and (external_spliter is None):
        raise(Exception("External validation but no exteranl spliter is given"))
    if (validation_mode != "external") and (external_spliter is not None):
        print("Validation mode is not external, external spliter is ignored")
        
    if (validation_mode == "external"):
        ((X, y), (Xv, yv)) = external_spliter(X, y)
    else:
        Xv, yv = None, None

    return X, y, Xv, yv
    
def to_parallelize(i, model, param_list, X, y, n_combis, n_tested,
                   Xv=None, yv=None, seeds=None, save_preds=False,
                   return_regr=None, verbose=True):
    """
    Create a regressor by feeding the i th parameter dict to the ModelWrapper model.
    Fit this regressor on X and y, applying external or internal splitting.
    validation_mode = external : the configuration is evaluated using a separated 
    validation split. Xv and yv must be provided
    validation_mode = internal : use the same validation set used by the model 
    (internally, use the same splitter). Xv and yv are ignored and the 
    model's splitter must be defined.
    validation_mode = oob : use the model's computed out of bag measures to 
    evaluate the configuration. Xv and yv are ignored and the model's splitter 
    must be defined.
    """
    simplefilter("ignore", category=ConvergenceWarning)
    current = n_tested + i
    if verbose or (current % 500) == 499:
        print(f"Fitting configuration{model.__class__}{model.prefix}, {current}")
    
    if seeds is not None:
        set_all_seeds(seeds[i])
        if verbose: print(f"Using SEED {seeds[i]}")
        
    oob = model.validation_mode == "oob"

    ptemp = copy.deepcopy(param_list[i])
    try: model.spliter = ptemp["spliter"]
    except: pass
    
    regr = model.make(model._params(ptemp))
    Xv, yv = inner_validation(model.validation_mode, X, y, Xv, yv, model)
    try:
        start = time.time()
        # We give X and y because validation splitting will either happen
        #internally or has already happened.
        regr.fit(X, y)
        stop = time.time()
        elapsed_time = stop - start
        #print(f"Trained in {elapsed_time}")
    except Exception as e:
        print(f"Error during the fit of combi {current} : ", e)

    try:
        # Evaluate the config.
        yvpred = model.predict_val(regr, Xv, oob=oob)
        res = mean_absolute_error(yv, yvpred)
        acc = CC(yv, yvpred)        
        if save_preds:
            pandas.DataFrame(yvpred).to_csv(
                model.validation_prediction_path(current), index=False)
        times = elapsed_time
    except Exception as e:
        print(f"Error during the evaluation of combi {current} : ", e)
        res = np.NaN
        acc = -1
        times = 0        
        
    if return_regr: return regr
    
    return res, acc, times

def parallelize(n_cpus, model, param_list, X, y, seeds=None, return_regr=False,
                verbose=True, restart=True, save_preds=False):
    if not restart:
        previous = model.load_results()
        n_tested = previous.shape[0]
    else:
        n_tested = 0
    n_combis = len(param_list)

    # If the model has self validation (DNN, CNN, GNN),
    # then outer validation does
    # nothing. Otherwise, this splits the validation set.
    X, y, Xv, yv = outer_validation(
        model.validation_mode, model.external_spliter, X, y)
    if n_cpus != 1:
        results = Parallel(n_jobs=n_cpus)(
            delayed(to_parallelize)(
                i, model, param_list, X, y,
                n_combis, n_tested,
                Xv=Xv, yv=yv, seeds=seeds, save_preds=save_preds,
                return_regr=return_regr, verbose=verbose)
            for i in range(n_combis))
    else:
        NUM_WORKERS = len(os.sched_getaffinity(0))
        if model.string() == "GNN":
            print(f"USING {NUM_WORKERS} WORKERS to parallelize GNN")
            torch.set_num_threads(NUM_WORKERS)

        # With torch, models are already trained in parallel!
        results = [to_parallelize(
            i, model, param_list, X, y,
            n_combis, n_tested,            
            Xv=Xv, yv=yv, seeds=seeds, save_preds=save_preds,
            return_regr=return_regr, verbose=verbose)
            for i in range(n_combis)]
    return results

def results_to_df(results, param_list, seeds=None, map_dict={}, cv=1, n_cpus=1):
    df = pandas.DataFrame(param_list)
    df['maes'] = [r[0] for r in results]
    df['accs'] = [r[1] for r in results]    
    df['times'] = [r[2] for r in results]
    if seeds is not None:
        df['seeds'] = seeds
    
    for k in map_dict.keys():        
        v = map_dict[k]
        # Handle the case of the structure parameters for the CNN
        if type(v) == dict:                           
            for v2 in v.keys():
                func = v[v2][0]
                if k not in df.columns:
                    df[v2] = [func(i) for i in df[v2]]
                else:
                    df[v2] = [func(i[v2]) for i in df[k]]

            if k in df.columns: del df[k]
        else:
            if k in df.keys(): df[k] = [v[0](i) for i in df[k]]
    return df

def recalibrate_parallel(predictions, model, best_params, Xtrain, ytrain, Xt, yt,
                         seed, times, ncpus, ntest, start, calibration_window=None,
                         shaps=None, n_shap=0, step=1, save=False):
    
    results = Parallel(n_jobs=ncpus)(
        delayed(recalibrate_parallel_)(
            i, model, best_params, Xtrain, ytrain, Xt, yt, seed, start, save=save,
            step=step, n_shap=n_shap, calibration_window=calibration_window)
        for i in range(0, ntest, step))

    for i, (preds, time_, shaps_) in enumerate(results):
        step_ = preds.shape[0]
        predictions[(i*step):(i*step)+step_] = preds
        times[i] = time_        
        if n_shap > 0:
            if len(yt.shape) > 1: nlabels = yt.shape[-1]
            else: nlabels = 1
            for h in range(nlabels):
                shaps[h, i:i+step_, :] = shaps_[h]
                
def recalibrate_parallel_(i, model,best_params, Xtrain, ytrain, Xt, yt, seed,start_,
                          calibration_window=None, n_shap=0, step=1, save=False):
    tf.keras.backend.clear_session()
    if (i % 25) == 0: print(f"Recalibrating for the day {start_+i+1}/{start_+yt.shape[0]}")
    print(f"RECALIBRATION {int(start_ / step) + int(i / step) + 1}/{int(start_ / step) + int(yt.shape[0] / step) + 1}")

    import warnings
    warnings.filterwarnings("ignore")    
    
    # Add the data to the training set
    Xtrain_ = np.concatenate((Xtrain, Xt[:i]))
    ytrain_ = np.concatenate((ytrain, yt[:i]))

    # Retrieve the day to predict
    Xi = Xt[i:i + step]
    yi = yt[i:i + step]
    if step == 1:
        Xi = Xi.reshape(1, -1)
        yi = yi.reshape(1, -1)

    # Select the calibration window
    Xtrain_, ytrain_ = handle_calibration_window(
        Xtrain_, ytrain_, calibration_window=calibration_window)

    # SET THE SEEDS
    if seed is not None: set_all_seeds(seed)
    
    # Refit the regressor
    start = time.time()
    regr = model.make(model._params(best_params))
    regr.fit(Xtrain_, ytrain_)
    stop = time.time()
    elapsed_time = stop - start
    time_ = elapsed_time    
    
    # Make the prediction
    predictions = model.predict_test(regr, Xi)    

    # Compute the shape value if specified
    shap_values = compute_shap_values(model,regr,Xtrain_, Xi, n_shap, method="mean")
    return predictions, time_, shap_values

def compute_shap_values(model, regr, X, Xi, n_shap, method="mean"):
    if n_shap > 0:
        def wrapper(x): return model.predict_test(regr, x)

        # Handle the date if the dataset got some
        if model.countries_to_predict_ != "not_graph":
            placeholder = X[0, model.psd_idx]
            X = copy.deepcopy(X)
            X[:, model.psd_idx] = np.zeros(X.shape[0])
        
        if method == "mean":
            # Use the mean value of each feature as the replacement
            data = X.mean(axis=0).reshape(1, -1)
        elif method == "yesterday":
            # Use feature of yesterday for the replacement. In a day to
            # day/recalibration framework, this corresponds to the last train sample
            data = X[-1, :].reshape(1, -1)

        # Handle the date if the dataset got some
        if model.countries_to_predict_ != "not_graph":
            data = data.astype(dtype=object)
            data[:, model.psd_idx] = placeholder
            
        explainer = shap.KernelExplainer(
            model=wrapper, data=data)
        warnings.filterwarnings("ignore")

        if Xi.shape[0] == 0:
            Xi = np.array(Xi.tolist(), dtype=object)
        shap_values = explainer.shap_values(X = Xi, nsamples = n_shap, silent=True,
                                            l1_reg="num_features(25)",
                                            gc_collect=True)
        return shap_values    
    
def handle_calibration_window(Xtrain_, ytrain_, calibration_window=None):
    ntrain = ytrain_.shape[0]
    if calibration_window is None: calibration_window = ntrain
    lower_bound = max(ntrain - calibration_window, 0)
    Xtrain_ = Xtrain_[lower_bound:ntrain]
    ytrain_ = ytrain_[lower_bound:ntrain]

    return Xtrain_, ytrain_

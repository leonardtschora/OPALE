import os, pandas, copy, ast, numpy as np, time
from ast import literal_eval as make_tuple
from tensorflow.keras import regularizers

import src.models.Splitter as SPL
from src.models.Splitter import MySplitter
import src.models.parallel_scikit as ps
import src.models.model_utils as mu

base_folder = os.environ['OPALE']

############################# MAIN functions
def run(**kwargs):
    params = default_params()
    params.update(kwargs)
    kwargs = params
    
    # TASKS
    GRID_SEARCH = kwargs["GRID_SEARCH"]
    RECALIBRATE = kwargs["RECALIBRATE"]

    # GENERAL PARAMS
    name = kwargs["name"]
    datasets = kwargs["datasets"]
    EPF = kwargs["EPF"]
    for_graph = kwargs["for_graph"]
    base_dataset_names = kwargs["base_dataset_names"]    
    n_val = kwargs["n_val"]    
    models = kwargs["models"]
    countries = kwargs["countries"]

    # GRID SEARCH PARAMS    
    fast = kwargs["fast"]
    n_combis = kwargs["n_combis"]
    restart = kwargs["restart"]
    n_rep = kwargs["n_rep"]    
    stop_after = kwargs["stop_after"]
    n_cpus = kwargs["n_cpus"]
    save_preds = kwargs["save_preds"]

    # RECALIBRATION PARAMS
    acc = kwargs["acc"]    
    n_shap = kwargs["n_shap"]
    start_rec = kwargs["start"]
    step = kwargs["step"]
    stop_rec = kwargs["stop"]
    calibration_window= kwargs["calibration_window"]
    filters = kwargs["filters"]
    inverted_filters = kwargs["inverted_filters"]
    recompute = kwargs["recompute"]     

    # Some init
    recalibration_times = pandas.DataFrame(columns=["country", "model", "times"])
    restart_changed = False
    for (dataset, base_dataset_name) in zip(datasets, base_dataset_names):
        for country in countries:
            for model in models:
                if type(model) is list:
                    model_params = model[1]
                    model = model[0]
                else:
                    model_params = {}

                if "n_cpus" in model_params:
                    n_cpus_ = model_params.pop("n_cpus")
                else:
                    n_cpus_ = n_cpus

                if "n_combis" in model_params:
                    n_combis_ = model_params.pop("n_combis")
                else:
                    n_combis_ = n_combis
                    
                start = time.time()
                if restart_changed:
                    restart_changed = False
                    restart = True
                
                if GRID_SEARCH:
                    for i in range(n_rep):
                        model_wrapper = run_grid_search(
                            name, dataset, model, country, base_dataset_name,
                            fast, EPF, n_combis_, restart, stop_after, n_cpus_,
                            n_val, model_params, i, save_preds)

                        if (n_rep > 1) and restart:
                            print(
                                "Step specified but wram start is not allowed.")
                            print("Disabling restart.")
                            restart_changed = True
                            restart = False
                    
                        df = model_wrapper.load_results()
                        best_params = model_wrapper.best_params(df)
                        print(f"LOSS = {best_params['maes']}")
                        print(f"TIME = {round((time.time() - start)  / 3600, ndigits=2)}h")
                    
                if RECALIBRATE:
                    pause = time.time()
                    total_time = run_recalibrate(
                        name, dataset, model, country, n_cpus_,
                        start_rec, stop_rec, step, n_shap, n_val,
                        base_dataset_name, calibration_window, model_params, EPF,
                        for_graph, filters, inverted_filters, recompute, acc)
                    recalibration_times = pandas.concat(
                        (recalibration_times,
                         pandas.DataFrame(
                             index=[0],
                             columns = ["country", "times", "models"],
                             data = [[country, total_time, model.string(model)]])),
                        ignore_index=True)
                    start = start - (time.time() - pause)

def run_grid_search(name, dataset, model, country, base_dataset_name, fast, EPF,
                    n_combis, restart, stop_after, n_cpus, n_val, model_params,
                    i, save_preds):
    spliter = MySplitter(n_val, shuffle = False)    
    model_wrapper = create_mw(country, dataset, model, name, EPF=EPF,
                              model_params=model_params, spliter=spliter)
    
    X, y = model_wrapper.load_train_dataset()
    n = X.shape[0]
    if base_dataset_name == dataset: load_from = None
    else: load_from = base_dataset_name
    search_space = get_search_space(
        model_wrapper, n, country, dataset=dataset, fast=fast,
        load_from=load_from, stop_after=stop_after)

    print("STARTING REPETITION : ", str(i))
    
    # This makes sure that all the models will have the same sampled parameters
    ps.set_all_seeds(1)
    param_list, seeds = ps.get_param_list_and_seeds(
        search_space, n_combis, model_wrapper=model_wrapper, restart=restart)
    results = ps.parallelize(n_cpus, model_wrapper, param_list, X, y,
                             seeds=seeds, restart=restart, save_preds=save_preds)
    df = ps.results_to_df(results, param_list, seeds=seeds, n_cpus=n_cpus,
                          map_dict=model_wrapper.map_dict(), cv=1)
    if not restart:
        # Dont use load results here because it will parse string as python objects!
        df = pandas.concat((pandas.read_csv(model_wrapper.results_path()), df),
                           ignore_index=True)
    df.to_csv(model_wrapper.results_path(), index=False)
    return model_wrapper

def run_recalibrate(name, dataset, model, country, n_cpus, start, stop, step,
                    n_shap, n_val, base_dataset_name, calibration_window,
                    model_params,EPF, for_graph, filters, inverted_filters,
                    recompute, acc):
    spliter = MySplitter(n_val, shuffle = True)
    model_wrapper = create_mw(country, dataset, model, name, EPF=EPF,
                              model_params=model_params, spliter=spliter)

    base_model_wrapper = copy.deepcopy(model_wrapper)
    base_model_wrapper.dataset_name = base_model_wrapper.dataset_name.replace(
        dataset, base_dataset_name)

    # This is for the Graph-Based models
    base_model_wrapper.load_train_dataset()
    model_wrapper.load_train_dataset()
    
    # Load the best params from this mw
    print(f"LOADING THE BEST CONFIGURATION FOR {model_wrapper.string()} FROM DATASET '{base_dataset_name}' WITH ATCs REPLACED BY '{model_wrapper.replace_ATC}'.")
    for f in filters:
        print(f"KEEPING ONLY THE VALUES {f}={filters[f]}\n")
    for f in inverted_filters:
        print(f"KEEPING ONLY THE VALUES {f}!={inverted_filters[f]}\n")        
    df = base_model_wrapper.load_results()
    best_params = base_model_wrapper.best_params(
        df, for_recalibration=True, acc=acc, recompute=recompute,
        filters=filters, inverted_filters=inverted_filters)
    
    # Recalibrate
    best_params["n_epochs"] = 1
    total_time = model_wrapper.recalibrate_epf(
        seed=best_params["seeds"], ncpus=n_cpus,
        calibration_window=calibration_window, filters=filters,
        inverted_filters=inverted_filters,
        best_params=best_params, n_shap=n_shap, start=start, stop=stop, step=step)
    return total_time

############################# Utility function for model wrappers
def create_mw(country, dataset, model, name, model_params={}, spliter=None,
              EPF="EPF"):
    dataset_ = f"{EPF}{dataset}"
    if EPF == "EPF": dataset_ += "_" + str(country)
    name_ = model.string(model) + "_" + name
    return model(name_, dataset_, country, spliter=spliter, **model_params)

def get_search_space(model_wrapper, n, country, stop_after=-1,
                     dataset="", fast=False, load_from=None):
    if load_from is None:
        return model_wrapper.get_search_space(
            country=country, n=n, fast=fast, stop_after=stop_after)

    # Load results from the original version
    base_model_wrapper = copy.deepcopy(model_wrapper)
    base_model_wrapper.dataset_name = base_model_wrapper.dataset_name.replace(
        dataset, load_from)
    return base_model_wrapper.load_results()

def flatten_dict(d):
    res = []
    for k in d.keys():
        if type(d[k]) is list:
            res += d[k]
        else:
            res += d[k].tolist()
            
    return res

def default_dataset_path(dataset_name):
    return os.path.join(base_folder, "data", "datasets")

def all_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "all.csv")

def extra_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "extra.csv")

def train_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "train.csv")

def val_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "val.csv")

def test_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "test.csv")

def folder(dataset_name):
    return os.path.join(base_folder, "data", "datasets")

def figure_folder(dataset_name):
    return os.path.join(base_folder, "figures", dataset_name)

def save_name(prefix, dataset_name):
    return prefix + "_" + dataset_name

def column_path(dataset_name, order_str):
    return os.path.join(folder(dataset_name), f"columns_{order_str}.txt")

def load_columns(dataset_name, order_str=""):
    with open(column_path(dataset_name, order_str), "r") as f:
        columns = f.read()
    columns = columns.split('", "')
    columns[0] = columns[0][2:]
    columns[-1] = columns[-1][:-3]
    if columns == ["y"]:
        columns = []
    return columns

def load_labels(dataset_name):
    with open(os.path.join(folder(dataset_name), "labels.txt"), "r") as f:
        columns = f.read()
    columns = columns.split('", "')
    columns[0] = columns[0][2:]
    columns[-1] = columns[-1][:-3]
    return columns    

def save_path(prefix, dataset_name):
    return os.path.join(folder(dataset_name), save_name(prefix, dataset_name))

def model_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_model"

def train_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_train_predictions.csv"

def val_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_val_predictions.csv"

def test_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_test_predictions.csv"

def test_recalibrated_prediction_path(model, replace_ATC):
    if replace_ATC != "":
        replace_ATC = "_" + replace_ATC
        
    path = os.path.join(os.environ["OPALE"], "data", "Predictions")
    return os.path.join(
        path, model + replace_ATC + "_test_recalibrated_predictions.csv")

def test_shape_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_test_shape_values.npy"

def test_recalibrated_shape_path(prefix, dataset_name, replace_ATC):
    if replace_ATC != "":
        replace_ATC = "_" + replace_ATC
    return save_path(prefix + replace_ATC, dataset_name) + "_test_recalibrated_shape_values.npy"

def all_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_all_predictions.csv"

def extra_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_extra_predictions.csv"

def past_label_columns(dataset, past_label):
    try:
        past_labels = [np.where(dataset.columns == c)[0][0] for c in past_label]
    except:
        past_labels = None
    
    return past_labels

def load_dataset(path, label, past_label):
    dataset = pandas.read_csv(path)
    labels = dataset[label]
    dataset.drop(columns=np.concatenate((np.array(label),
                                         np.array(["period_start_date"]))),
                 inplace=True)
    past_label_idx = past_label_columns(dataset, past_label)
    return dataset, labels, past_label_idx

def load_dataset_(dataset, labels, label):
    X = dataset.values
    y = labels.values
    if len(label) == 1:
        y = y.reshape(1, -1).ravel()
    return X, y

def neurons_per_layer_to_string(npl):
    return str(npl)

def neurons_per_layer_from_string(res):
    return make_tuple(res)

def spliter_to_string(spliter):
    return str(spliter)

def spliter_from_string(res):
    clas, val, shuffle = res.split("_")
    obj = getattr(SPL, clas)
    val = float(val)
    shuffle = bool(shuffle)
    return obj(val, shuffle=shuffle)

def filter_size_to_string(fs):
    return str(fs)

def filter_size_from_string(res):    
    return make_tuple(res)

def dilation_rate_to_string(dr):
    return str(dr)

def dilation_rate_from_string(res):
    return make_tuple(res)

def activity_regularizer_to_string(activity_regularizer):
    if activity_regularizer is not None:
        res = activity_regularizer.get_config()
    else: res = ""
    return res

def activity_regularizer_from_string(res):
    if res == "": return  None    
    if pandas.isna(res): return None

    res = ast.literal_eval(res)
    typ = [k for k in res.keys()]
    if len(typ) == 1:
        typ = typ[0]
    else:
        typ = "L1L2"

    return getattr(regularizers, typ)(**res)

def load_results(path, map_dict):
    df = pandas.read_csv(path)
    for k in map_dict.keys():
        # Handle nested params
        if type(map_dict[k]) == dict:
            for k2 in map_dict[k].keys():
                _, f = map_dict[k][k2]
                df[k2] = [f(i) for i in df[k2]]
        else:
            _, f = map_dict[k]
            if k in df.keys(): df[k] = [f(i) for i in df[k]]
    return df        

###################### DEFAULT RUN PARAMS
def default_params():
    return {
        # TASKS
        "GRID_SEARCH" : False,
        "LOAD_AND_TEST" : False,
        "RECALIBRATE" : False,   

        # GENERAL PARAMS
        "name" : "TSCHORA",
        "datasets" : ("AllEuropeGNN",),
        "base_dataset_names" : ("AllEuropeGNN",),
        "EPF" : "",
        "for_graph" : True,
        "n_val" : 365,
        "n_cpus" : 1,    
        "filters" : {},
        "inverted_filters" : {},
        "models" : ([], []),
        "countries" : ("FR", ),
    
        # GRID SEARCH PARAMS
        "fast" : False,
        "n_combis" : 1,
        "restart" : False,
        "n_rep" : 1,
        "stop_after" : 250,
        "save_preds" : False,
        
        # RECALIBRATION PARAMS
        "acc" : False,
        "recompute" : False,    
        "n_shap" : 0,
        "start" : 0,
        "step" : 30,
        "stop" : 731,        
        "calibration_window" : 1440,
    }

import numpy as np, pandas, matplotlib.pyplot as plt, matplotlib, copy

import matplotlib.ticker as tck
import matplotlib.patches as mpatches

from src.analysis.metrics import CC, mae, smape, DM
from src.models.gnn_wrapper import GNNWrapper
from src.models.splitter import MySplitter

def load_real_prices():
    nval = 365
    spliter = MySplitter (nval, shuffle=False)
    model_wrapper = GNNWrapper("TSCHORA", "AllEuropeGNN", spliter=spliter,
                        known_countries=["CH", "GB"], countries_to_predict="all")
    _, Y = model_wrapper.load_test_dataset()
    return Y

def load_forecasts(models, nd, ny):
    nv = len(models.keys())
    nm = np.sum([len(models[k]) for k in models.keys()])
    predictions = np.zeros((nm, nd, ny))
    model_wrappers = []
    k = 0
    for i, replace_atc in enumerate(models.keys()):
        for j, model in enumerate(models[replace_atc]):
            model_wrapper = model(
                f"{model.string(model)}_TSCHORA", "AllEuropeGNN",
                flow_estimation = replace_atc,
                spliter = MySplitter(365, shuffle=False),
                known_countries = ["CH", "GB"],
                countries_to_predict = "all")
            model_wrappers.append(model_wrapper)
            try:
                predictions[k] = pandas.read_csv(
                    model_wrapper.test_recalibrated_prediction_path()).values[:nd, :ny]
            except FileNotFoundError:
                print(f"No file at {model_wrapper.test_recalibrated_prediction_path()}")
            k = k + 1

    return predictions, model_wrappers, nv, nm

def compute_metrics(Y, predictions, nm, model_wrappers, metrics=(CC, mae, smape)):
    results = np.zeros((nm, len(metrics)))
    for i in range(nm):
        for j, metric in enumerate(metrics):
            results[i, j] = metric(Y, predictions[i])

    df = pandas.DataFrame(
        columns = [m.__name__ for m in metrics],
        index = [m.string() + m.replace_ATC for m in model_wrappers],
        data = results)            
    return df

def compute_DM_tests_(Y, predictions, nm, model_wrappers):
    # Compute DM tests
    pvalues_smape = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            if i == j:
                pvalues_smape[i, j] = np.nan         
            else:
                pvalues_smape[i, j] = DM(Y, predictions[i], predictions[j],
                                         norm="mae")

    df_smape = pandas.DataFrame(
        columns = [m.string() + m.replace_ATC for m in model_wrappers],
        index = [m.string() + m.replace_ATC for m in model_wrappers],
        data=pvalues_smape)
                
    return df_smape

def reshape_DM_tests_version(df, models, model_wrappers, nm, nv):
    new_shape = np.zeros((nm, nv))
    idx = []
    for (i, model) in enumerate(models[""]):
        indices = [m.string() + m.replace_ATC
                   for m in model_wrappers if type(m) is model]
        idx += indices
        data = df.loc[indices, indices].values
        data = np.where(np.isnan(data), 1, data)
    
        start = i * nv
        stop = start + nv
        new_shape[start:stop, :] = data
    df_comp = pandas.DataFrame(
        columns=[m.replace_ATC
                 for m in np.array(model_wrappers)[np.arange(0,nm,int(nm/nv))]],
        index=idx, data=new_shape)
    return df_comp

def reshape_DM_tests_models(df, models, model_wrappers, nm, nv):
    new_shape = np.zeros((nm, nv))
    idx = []
    for (i, version) in enumerate(models.keys()):
        indices = [m.string() + m.replace_ATC
                   for m in model_wrappers if m.replace_ATC == version]
        idx += indices
        data = df.loc[indices, indices].values
        data = np.where(np.isnan(data), 1, data)
    
        start = i * nv
        stop = start + nv
        new_shape[start:stop, :] = data
    df_comp = pandas.DataFrame(
        columns=[m.string(m) for m in models[""]], index=idx, data=new_shape)
    return df_comp

def compute_DM_tests(Y, predictions, nm, nv, model_wrappers, models):
    pvalues = compute_DM_tests_(Y, predictions, nm, model_wrappers)
    df_comp = reshape_DM_tests_version(pvalues, models, model_wrappers, nm, nv)
    df_comp_models = reshape_DM_tests_models(pvalues, models, model_wrappers,
                                             nm,len(models[""]))
    df_all = df_comp.join(df_comp_models)
    return round(1 - df_all, ndigits=3)    

def metric_by_zone(Y, zones, predictions, metric, model_wrappers):
    results = np.zeros((len(predictions), int(Y.shape[1]/24)))
    for i, ypred in enumerate(predictions):
        results[i, :] = metric(Y, ypred, mean=False).reshape(-1, 24).mean(axis=1)

    df = pandas.DataFrame(
        index=zones,
        columns=[m.string() + m.replace_ATC for m in model_wrappers],
        data=results.transpose())        
    return df

def sort_by_version(df):
    df.loc[:, "version"] = [map_version(v) for v in df.index]
    df.loc[:, "model"] = [map_model(m) for m in df.index]    
    df = df.sort_values(["version", "model"])
    df.drop(columns=["version", "model"], inplace=True)
    return df

def sort_by_models(df, by_col=False):
    if by_col:
        df = df.transpose()
    df.loc[:, "version"] = [map_version(v) for v in df.index]
    df.loc[:, "model"] = [map_model(m) for m in df.index]    
    df = df.sort_values(["model", "version"])
    df.drop(columns=["version", "model"], inplace=True)
    if by_col:
        df = df.transpose()
    return df

def map_version(v):
    v = v.split("NN")[1]
    if v in ("atc", "A"): return 0
    if v == "lin": return 1
    if v == "lsq": return 2
    if v == "cmb": return 3
    if v == "os": return 4

def map_model(m):
    tm = m.split("NN")[0] + "NN"
    if "DNN" in tm: return 1
    if "CNN" in tm: return 2
    if "GNN" in tm: return 3

def latex_zone(z):
    if len(z) == 2:
        return z
    else:
        if "IT" in z:
            number = z[2:]
            zone = "IT"
        else:
            number = z[-1]
            zone = z[-3:-1]
        return zone + "-" + number    
    
def plot_metric_by_zones(df, metric):
    df = sort_by_models(df, by_col=True).transpose()    
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    im = ax.imshow(df, cmap="viridis_r");

    xt, yt = df.values.shape
    
    ax.set_xticks(range(yt))
    ax.set_xticklabels([latex_zone(l) for l in df.columns], rotation=45)
    
    ax.set_yticks([i for i in range(xt)])
    ax.set_yticklabels(df.index)
    plt.colorbar(im)
    plt.title(f"{metric.__name__} for all zones")
    plt.show()    

def get_zones(model_wrappers):
    model_wrapper = model_wrappers[0]
    model_wrapper.load_train_dataset()
    zones = model_wrapper.countries_to_predict
    return zones

def compute_DM_tests_by_zones(Y, predictions, nz, model_wrappers):
    # Compute DM tests
    nm = len(model_wrappers)
    pvalues_smape = np.zeros((nm, nm, nz))
    for i in range(nm):
        for j in range(nm):
            if i == j:
                pvalues_smape[i, j, :] = np.nan         
            else:
                for z in range(nz):
                    zone = [z * 24 + k for k in range(24)]
                    pvalues_smape[i, j, z] = DM(
                        Y[:,zone], predictions[i][:, zone], predictions[j][:, zone])
                            
    return pvalues_smape

def plot_DM_results_by_zone(model_wrappers, zones, m):
    versions = np.array([m.replace_ATC for m in model_wrappers])
    models = np.array([m.string() for m in model_wrappers])

    nm = len(np.unique(models))
    nv = int(len(models) / nm)
    umodels = models[range(0, nm)]
    uversions = versions[range(0, len(versions), nm)]
    
    xindices = np.array([d for d in range(33)])
    results = np.zeros((len(umodels) * (len(uversions) - 1), len(zones)))
    c = 0
    resulting_models = []
    for i, model in enumerate(umodels):
        for j, version in enumerate(uversions):
            if version != "":
                k = np.where(np.logical_and(
                    versions == version, models == model))[0][0]
                ref = np.where(np.logical_and(
                    versions == "", models == model))[0][0]
                #forward = m[k, ref, :]
                backward = copy.deepcopy(m[ref, k, :])

                res = np.zeros(backward.shape, dtype=np.float64)
                res[backward < 0.05] = 1
                res[backward > 0.95] = -1
                results[c, :] = res
                resulting_models += [version]
                c +=1

    with matplotlib.rc_context({ "text.usetex" : True,
                                 "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc}",
                             "font.family" : ""}):
        params = {"fontsize" : 25, "fontsize_labels" : 18, "linewidth" : 3}
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
                
        im = ax.imshow(results, vmin=-1, vmax=1, cmap="RdYlGn")
    
        ax.set_xticks(xindices)
        zones = [latex_zone(z) for z in zones]
        ax.set_xticklabels(zones, rotation=45, fontsize=params["fontsize_labels"])

        yindices = np.array([d for d in range(results.shape[0])])
        ax.set_yticks(yindices)
        ax.set_yticklabels(resulting_models, fontsize=params["fontsize_labels"])

        ax.set_xticks(xindices - 0.52, minor=True)
        ax.set_yticks(yindices - 0.51, minor=True)        
        ax.grid("on", which="minor")

        for i, model in enumerate(umodels[::-1]):
            y = i * (len(uversions) - 1)
            plt.annotate(model,
                         xy=[0, 0],
                         xytext=(-0.065, (y + 2 * len(umodels)/3)/results.shape[0]),
                         rotation=90, fontsize=params["fontsize"],
                         va="center", ha="center", textcoords="axes fraction")

            if i != 0:
                yarrow = y/results.shape[0]
                plt.annotate("",
                             xy=[1 + 0.005, yarrow],
                             xycoords="axes fraction",
                             xytext = [-0.075, yarrow],                         
                             textcoords="axes fraction",                          
                             arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                         "linewidth" : 4, "color" : "k"})
                
        cmap = plt.get_cmap("RdYlGn")
        red_patch = mpatches.Patch(color=cmap(0), label='Significant deterioration')
        yellow_patch = mpatches.Patch(color=cmap(0.5), label='No conclusion')
        green_patch=mpatches.Patch(color=cmap(0.99),label='Significant improvement')
    
        ax.legend(handles = [green_patch, yellow_patch, red_patch],
                  bbox_to_anchor=(0.2, 0.78), bbox_transform=fig.transFigure,
                  fontsize=params["fontsize_labels"], ncol=3, loc="lower left")
        plt.suptitle("DM test results between models using $A$ and models using different \\textbf{F}", fontsize=params["fontsize"], y=0.9)
        plt.show()
    return pandas.DataFrame(index=resulting_models, columns=zones, data=results)

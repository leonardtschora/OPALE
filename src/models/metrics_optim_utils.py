import pandas, numpy as np, matplotlib.pyplot as plt, copy
from work.models.Splitter import MySplitter
from work.models.DNNWrapper import DNNWrapper
from work.models.RFR import RFR
from work.models.Feature import NaiveRecalibration
from work.models.CNNWrapper import CNNDropWrapper
from work.models.GNNWrapper import NodeGNNWrapper
from work.analysis.metrics_utils import DM
from work.analysis.evaluate import mae, smape, rmae, dae, cmap, ACC, rACC, cmap_2
import matplotlib.gridspec as gridspec
import matplotlib.ticker as tck
import matplotlib.patches as mpatches
from work.flux_utils import *

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

def load_real_prices():
    nval = 365
    spliter = MySplitter(nval, shuffle=False)
    model_wrapper = RFR("RF_TSCHORA", "AllEuropeGNN", spliter=spliter,
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
                replace_ATC = replace_atc,
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

def metric_by_zone(Y, predictions, metric):
    results = np.zeros((len(predictions), int(Y.shape[1]/24)))
    for i, ypred in enumerate(predictions):
        results[i, :] = metric(Y, ypred, mean=False).reshape(-1, 24).mean(axis=1)

    return results

def plot_metric_by_zones(df, metric):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    im = ax.imshow(df, cmap="viridis_r");

    xt, yt = df.values.shape
    
    ax.set_xticks(range(yt))
    ax.set_xticklabels(df.columns, rotation=45)
    
    ax.set_yticks([i for i in range(xt)])
    ax.set_yticklabels(df.index)
    plt.colorbar(im)
    plt.title(f"{metric.__name__} for all zones")
    plt.show()    

def compute_metrics_O(Y, predictions, nm, naive_forecasts=None,
                    metrics=(ACC, mae, dae, smape)):
    results = np.zeros((nm, len(metrics)))
    for i in range(nm):
        for j, metric in enumerate(metrics):
            if metric not in (rmae, rACC):
                results[i, j] = metric(Y, predictions[i])
            else:
                results[i, j] = metric(Y, predictions[i], naive_forecasts)
    return results

def compute_DM_tests(Y, predictions, nm, model_wrappers):
    # Compute DM tests
    pvalues_mae = np.zeros((nm, nm))
    pvalues_smape = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            if i == j:
                pvalues_mae[i, j] = np.nan
                pvalues_smape[i, j] = np.nan         
            else:
                pvalues_mae[i, j] = DM(Y, predictions[i], predictions[j],
                                       norm="mae", version="multivariate")
                pvalues_smape[i, j] = DM(Y, predictions[i], predictions[j],
                                         norm="smape", version="multivariate")

    df_maes = pandas.DataFrame(
        columns = [m.string() + m.replace_ATC_string() for m in model_wrappers],
        index = [m.string() + m.replace_ATC_string() for m in model_wrappers],
        data=pvalues_mae)
    df_smape = pandas.DataFrame(
        columns = [m.string() + m.replace_ATC_string() for m in model_wrappers],
        index = [m.string() + m.replace_ATC_string() for m in model_wrappers],
        data=pvalues_smape)
                
    return df_maes, df_smape

def compute_DM_tests_by_zones(Y, predictions, nz, model_wrappers, stat=False):
    # Compute DM tests
    nm = len(model_wrappers)
    pvalues_mae = np.zeros((nm, nm, nz))
    pvalues_smape = np.zeros((nm, nm, nz))
    for i in range(nm):
        for j in range(nm):
            if i == j:
                pvalues_mae[i, j, :] = np.nan
                pvalues_smape[i, j, :] = np.nan         
            else:
                for z in range(nz):
                    zone = [z * 24 + k for k in range(24)]
                    pvalues_mae[i, j, z] = DM(
                        Y[:, zone], predictions[i][:, zone], predictions[j][:,zone],
                        norm="mae", version="multivariate", stat=stat)
                    pvalues_smape[i, j, z] = DM(
                        Y[:,zone], predictions[i][:, zone], predictions[j][:, zone],
                        norm="smape", version="multivariate", stat=stat)
                            
    return pvalues_mae, pvalues_smape

def reshape_DM_tests_version(df, models, model_wrappers, nm, nv):
    new_shape = np.zeros((nm, nv))
    idx = []
    for (i, model) in enumerate(models[""]):
        indices = [m.string() + m.replace_ATC_string()
                   for m in model_wrappers if type(m) is model]
        idx += indices
        data = df.loc[indices, indices].values
        data = np.where(np.isnan(data), 1, data)
    
        start = i * nv
        stop = start + nv
        new_shape[start:stop, :] = data
    df_comp = pandas.DataFrame(
        columns=[m.replace_ATC_string().split("\_")[1]
                 for m in np.array(model_wrappers)[np.arange(0,nm,int(nm/nv))]],
        index=idx, data=new_shape)
    return df_comp

def reshape_DM_tests_models(df, models, model_wrappers, nm, nv):
    new_shape = np.zeros((nm, nv))
    idx = []
    for (i, version) in enumerate(models.keys()):
        indices = [m.string() + m.replace_ATC_string()
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

def plot_DM_tests_reshaped_version(df, models, model_wrappers):
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
    axes = axes.ravel()
    for (i, model) in enumerate(models[""]):
        ax = axes[i]
        indices = [m.string() + m.replace_ATC_string()
                   for m in model_wrappers if type(m) is model]
        data = df.loc[indices, :].values
        data = np.where(np.isnan(data), 1, data)
        im = ax.imshow(data, cmap=cmap_2(), vmin=0, vmax=0.051)
        
        tcklabel = [m.replace_ATC
                    for m in model_wrappers if type(m) is model]
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(tcklabel, rotation=45)
        
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(tcklabel)
        
        ax.set_title(model.string(model))
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05)
    cbar.ax.set_xlabel("Pvalue of the DM test")
    plt.suptitle("PValues for all models")
    plt.show()

def plot_DM_tests_reshaped_models(df, models, model_wrappers):
    factor = 1
    fig = plt.figure(figsize=(factor * 19.2, factor * 10.8))
    gs0 = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[0])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs0[1])

    ax1 = fig.add_subplot(gs1[:, 0])
    ax2 = fig.add_subplot(gs1[:, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs1[:, 2], sharex=ax1)

    ax4 = fig.add_subplot(gs2[:, 1:3])
    ax5 = fig.add_subplot(gs2[:, 3:5], sharex=ax4)
    axes = np.array([ax1, ax2, ax3, ax4, ax5])
    
    for (i, version) in enumerate(models.keys()):
        ax = axes[i]
        indices = [m.string() + m.replace_ATC_string()
                   for m in model_wrappers if m.replace_ATC == version]
        data = df.loc[indices, :].values
        data = np.where(np.isnan(data), 1, data)
        im = ax.imshow(data, cmap=cmap_2(), vmin=0, vmax=0.051)
        
        tcklabel = [m.string(m) for m in models[""]]
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels(tcklabel, rotation=45)
        
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(tcklabel)
        
        ax.set_title(version)
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05)
    cbar.ax.set_xlabel("Pvalue of the DM test")
    plt.suptitle("PValues for all models")
    plt.show()

def plot_diff_zones_versions(df, metric):    
    models = np.array([m.split("\_")[0] for m in df.index])
    nm = len(np.unique(models))
    nv = int(len(models) / nm)
    models = models[range(0, len(models), nv)]
    
    versions = np.array(["\_" + m.split("\_")[1] for m in df.index])
    versions = versions[range(nv)]
    
    diffs = []
    for m in models:
        mod = df.loc[[f"{m}{v}" for v in versions if v != "\_A"]].values
        atc = df.loc[f"{m}\_A"].values
        difs = mod - atc
        diffs += [difs]

    vabs = np.array([np.abs(d).max() for d in diffs]).max()
    fig, axes = plt.subplots(
        len(models),1,figsize=(19.2, 10.8),sharex=True, gridspec_kw={"wspace": 0.0})
    if metric == 'ACC':
        cmap = "seismic_r"
    else:
        cmap = "seismic"
    tcklabels = [f"{v} - atc" for v in versions if v != "\_A"]
    for i, ax in enumerate(axes):    
        im = ax.imshow(diffs[i], cmap=cmap, vmin=-vabs, vmax=vabs)
        ax.set_yticks(range(len(versions) - 1))
        ax.set_yticklabels(tcklabels)
        ax.set_title(models[i])

    ax.set_xticks(range(33))
    ax.set_xticklabels(df.columns, rotation=45)
    fig.colorbar(im, ax=axes)
    plt.suptitle(
        f"{metric} variation for different zones, comapared with ATCs")
    plt.show()


def plot_diff_zones_models(Y, predictions, zones, metric, model_wrappers):
    results = metric_by_zone(Y, predictions, metric)
    df = pandas.DataFrame(
        columns=zones,
        index=[m.string() + m.replace_ATC_string() for m in model_wrappers],
        data=results)
    df = sort_by_models(df)    

    models = np.array([m.string() for m in model_wrappers])[
        [d for d in range(0, 20,5)]]
    versions=np.array([m.replace_ATC_string() for m in model_wrappers])[
        [d for d in range(0, 20,4)]]

    #ms = [i.split('\_')[0] for i in df.index]
    #vs = [i.split('\_')[1] for i in df.index]    
    
    diffs = []
    for v in versions:
        mod = df.loc[[f"{m}{v}" for m in models if m != "RF"]].values
        atc = df.loc[f"RF{v}"].values
        difs = mod - atc
        diffs += [difs]

    vabs = np.array([np.abs(d).max() for d in diffs]).max()
    fig, axes = plt.subplots(
        len(versions),1,figsize=(19.2, 10.8),sharex=True,gridspec_kw={"wspace":0.0})
    if metric.__name__ == 'ACC':
        cmap = "seismic_r"
    else:
        cmap = "seismic"
    tcklabels = [f"{m} - RF" for m in models if m != "RF"]
    for i, ax in enumerate(axes):    
        im = ax.imshow(diffs[i], cmap=cmap, vmin=-vabs, vmax=vabs)
        ax.set_yticks(range(len(models) - 1))
        ax.set_yticklabels(tcklabels)
        ax.set_title(versions[i])

    ax.set_xticks(range(33))
    ax.set_xticklabels(zones, rotation=45)
    fig.colorbar(im, ax=axes)
    plt.suptitle(
        f"{metric.__name__} variation for different zones, comapared with ATCs")
    plt.show()
    
def plot_DM_stats_by_zone(model_wrappers, zones, m):
    versions = np.array([m.replace_ATC_string() for m in model_wrappers])
    models = np.array([m.string() for m in model_wrappers])
    
    umodels = np.unique(models)
    uversions = np.unique(versions)
    
    xindices = range(33)
    colors = np.array(["r", "b", "c", "m"])
    markers = np.array(["o", "v", "x", "s"])
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    mins = []
    maxs = []
    for i, model in enumerate(umodels):
        for j, version in enumerate(uversions):
            if version != "\_A":
                k = np.where(np.logical_and(
                    versions == version, models == model))[0][0]
                ref=np.where(np.logical_and(
                    versions == "\_A", models == model))[0][0]
                forward = m[k, ref, :]
                backward = m[ref, k, :]
                color = colors[np.where(model == umodels)[0][0]]
                marker = markers[np.where(version == uversions)[0][0] - 1]
                mins += [np.min(backward)]
                maxs += [np.max(backward)]
                ax.plot(xindices, backward, c=color, marker=marker,
                        label=f"{model}{version}", linewidth=0.75)
    ax.plot(xindices, [1.645 for i in xindices], linestyle="--", c="k")
    ax.plot(xindices, [-1.645 for i in xindices], linestyle="--", c="k")
    ax.plot(xindices, [0 for i in xindices], c="k")
    
    ax.set_xticks(xindices)
    ax.set_xticklabels(zones, rotation=45)

    ax.set_xlim([-0.5, len(xindices) - 0.5])
    ax.set_ylim([np.min(mins), np.max(maxs)])
    
    ax.yaxis.set_minor_locator(tck.MultipleLocator(0.2))

    ax.grid("on", which="major", linestyle="-", linewidth=1, color="k")
    ax.grid("on", which="minor", linestyle="--", linewidth=0.5, color="k")
    
    ax.legend()
    plt.show()


def plot_DM_results_by_zone(model_wrappers, zones, m, params):
    versions = np.array([m.replace_ATC_string() for m in model_wrappers])
    models = np.array([m.string() for m in model_wrappers])

    nm = len(np.unique(models))
    nv = int(len(models) / nm)
    umodels = models[range(0, nm)]
    uversions = versions[range(0, len(versions), nm)]
    
    xindices = np.array([d for d in range(33)])
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    results = np.zeros((len(umodels) * (len(uversions) - 1), len(zones)))
    c = 0
    resulting_models = []
    for i, model in enumerate(umodels):
        for j, version in enumerate(uversions):
            if version != "\_A":
                k = np.where(np.logical_and(
                    versions == version, models == model))[0][0]
                ref = np.where(np.logical_and(
                    versions == "\_A", models == model))[0][0]
                #forward = m[k, ref, :]
                backward = copy.deepcopy(m[ref, k, :])

                res = np.zeros(backward.shape, dtype=np.float64)
                res[backward < 0.05] = 1
                res[backward > 0.95] = -1
                results[c, :] = res
                resulting_models += ["F" + version.split("\_")[1]]
                c +=1

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
        plt.annotate(map_model_name(model),
                     xy=[0, 0],
                     xytext=(-0.065, (y + 2 * len(umodels)/3) / results.shape[0]),
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
    green_patch = mpatches.Patch(color=cmap(0.99), label='Significant improvement')
    
    ax.legend(handles = [green_patch, yellow_patch, red_patch],
              bbox_to_anchor=(0.2, 0.78), bbox_transform=fig.transFigure,
              fontsize=params["fontsize_labels"], ncol=3, loc="lower left")
    plt.suptitle("DM test results between models using $A$ and models using different \\textbf{F}", fontsize=params["fontsize"], y=0.9)
    plt.show()
    return pandas.DataFrame(index=resulting_models, columns=zones, data=results)

def plot_shap_move_by_zone(df, params, res):
    zones = df.columns
    versions = np.array([d.split("NN")[1] for d in df.index])
    versions_l = np.array([map_Flux_versions(d) for d in versions])

    models = np.array([m.split(v)[0] for m, v in zip(df.index, versions)])
    models_l = np.array([map_model_name(m) for m in models])

    nm = len(np.unique(models))
    nv = int(len(models) / nm)
    umodels = models[range(0, len(versions), nv)]
    uversions = versions[range(0, nv)]
    
    xindices = np.array([d for d in range(len(zones))])
    
    fig, ax = plt.subplots(figsize=(19.2, 10.8))    
    vabs = 0.5 * np.abs(df.values).max()
    im = ax.imshow(df.values, cmap="RdYlGn", vmin=-vabs, vmax=vabs);

    """
    for i, zone in enumerate(res.columns):
        for j, model in enumerate(res.index):
            xt = i
            yt = j            
            if res.loc[model, zone] == 1:                
                ax.text(i, j, "+", fontsize=params["fontsize_labels"],
                        ha="center", va="center")
            if res.loc[model, zone] == -1:                
                ax.text(i, j, "-", fontsize=params["fontsize_labels"],
                ha="center", va="center")
    """
     
    ax.set_xticks(xindices)
    zones = [latex_zone(z) for z in zones]
    ax.set_xticklabels(zones, rotation=45, fontsize=params["fontsize_labels"])

    yindices = np.array([d for d in range(df.values.shape[0])])
    ax.set_yticks(yindices)
    resulting_models = versions_l
    ax.set_yticklabels(resulting_models, fontsize=params["fontsize_labels"])

    ax.set_xticks(xindices - 0.52, minor=True)
    ax.set_yticks(yindices - 0.51, minor=True)        
    ax.grid("on", which="minor")

    for i, model in enumerate(umodels[::-1]):
        y = i * len(uversions)
        plt.annotate(map_model_name(model),
                     xy=[0, 0],
                     xytext=(-0.065, (y + 2 * len(umodels)/3) / df.values.shape[0]),
                     rotation=90, fontsize=params["fontsize"],
                     va="center", ha="center", textcoords="axes fraction")

        if i != 0:
            yarrow = y/df.values.shape[0]
            plt.annotate("",
                         xy=[1 + 0.005, yarrow],
                         xycoords="axes fraction",
                         xytext = [-0.075, yarrow],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 4, "color" : "k"})

    cbar = plt.colorbar(im, ax=ax, location="top", shrink=0.5, pad=0.01)
    cbar.ax.tick_params(labelsize=params["fontsize_labels"], pad=-30)
    plt.suptitle("Difference of contribution between \\textbf{F} and \\textbf{A} (\%)", fontsize=params["fontsize"], y=0.87)
    plt.show()    
    

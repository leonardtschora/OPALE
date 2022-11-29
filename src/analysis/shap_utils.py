import numpy as np, pandas, matplotlib.pyplot as plt, matplotlib, copy

import matplotlib.ticker as tck
import matplotlib.patches as mpatches

from src.analysis.metrics_utils import *

def normalize_shap(data):
    # Total contribution for each predicted label = 100%
    data = np.abs(data)
    per_label = data.sum(axis=2)    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :] /= per_label[i, j]
            
    data *= 100            
    return data

def mw(model, version):
    model_wrapper = model(
        f"{model.string(model)}_TSCHORA", "AllEuropeGNN",
        flow_estimation = version, spliter = MySplitter(365, shuffle=False),
        known_countries = ["CH", "GB"], countries_to_predict = "all")
    X, Y = model_wrapper.load_train_dataset()
    return model_wrapper

def by_zone_to_predict_features(values, model_wrapper,
                                connections=True, normalize=False):
    zones_label = model_wrapper.countries_to_predict
    zones = model_wrapper.countries_to_predict + model_wrapper.known_countries
    features = ["consumption", "production", "renewables_production", "price"]
    if connections: features += ["atc"]
    
    results = pandas.DataFrame(index=zones_label, columns=features,
                               data=np.zeros((len(zones_label), len(features))))
    # Fill each destination
    for j, z in enumerate(zones_label):
        xindices = np.array([i for i in range(j * 24, (j + 1) * 24)]).reshape(-1, 1)

        # Go through each departure zones
        for zprime in zones:

            # Process feature per feature
            for (i, f) in enumerate(features):
                if f == "atc":
                    keys = [f"{zprime}_{zpp}" for zpp in zones]
                    count_con = 0
                    to_add = 0
                    for k in keys:
                        inds = model_wrapper.edges_columns_idx[k]
                        if len(inds) > 0:
                            to_add += np.sum(values[xindices, inds])
                            count_con += 1                            
                    d = 1
                    if normalize: d = 1 * count_con
                    results.loc[z, f] += to_add / d                    
                else:
                    yindices = model_wrapper.country_idx[zprime][24*i:24*(i+1)]
                    d = 1
                    if normalize: d = len(yindices)
                    results.loc[z, f] += np.sum(values[xindices, yindices]) / d
        results.loc[z] = 100 * results.loc[z] / results.loc[z].sum()
    results.rename({"atc" : "flows"}, inplace=True)
    return results

def plot_features_summary(df):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    bar_width = 0.8
    cmap = plt.get_cmap("viridis")
    previous = np.zeros(len(df.index))
    for i, feature in enumerate(df.columns):
        xindices = [j for j in range(len(df.index))]
        ci = i / (len(df.columns) - 1)
        
        ax.bar(xindices, df.loc[:, feature], bottom=previous,
               color=cmap(ci), width=bar_width, label=feature)
        previous += df.loc[:, feature]

    ax.grid("on", axis="y", which="major", linestyle="-", linewidth=1, color="k")
    ax.grid("on", which="minor", linestyle="--", linewidth=0.5, color="k")
    ax.set_xlabel("Models")
    ax.set_ylabel("Contribution \%")

    ax.set_ylim([0, 100])
    ax.set_xlim([0 - 0.5, len(df.index) - bar_width / 2])
    
    ax.set_xticks([j for j in range(len(df.index))])
    ax.set_xticklabels(df.index, rotation=45)
    ax.yaxis.set_minor_locator(tck.MultipleLocator(1))

    ax.set_title("Contribution for each zone, summed by features")
    plt.legend()
    plt.show()

def load_shaps(models):
    results = {}
    for k in models:
        for model in models[k]:
            model_wrapper = mw(model, k)
            path = model_wrapper.test_recalibrated_shape_path()
            shaps = normalize_shap(np.load(path))

            key = f"{model_wrapper.string()}{k}"
            results[key] = by_zone_to_predict_features(
                shaps.mean(axis=1), model_wrapper)

    return results

def contribution_by_features(models, all_shaps):
    df = pandas.DataFrame(columns=all_shaps["DNN"].columns,
                          index=list(all_shaps.keys()),
                          dtype=np.float64)
    for k in models:
        for model in models[k]:
            key = f"{model.string(model)}{k}"
            df.loc[key, :] = all_shaps[key].mean()
        
    df.columns = ["C", "G", "R", "P", "F"]
    dividers = np.array([35, 35, 35, 35, 126])
    df  = (df / dividers)
    for i in df.index:
        df.loc[i] = 100 * df.loc[i] / df.loc[i].sum()

    return df

def compute_differences(all_shaps, models):
    df = pandas.DataFrame(index=[d for d in all_shaps.keys() if len(d) > 7],
                          columns=all_shaps["DNN"].index,
                          dtype=np.float64)
    for k in models:
        for model in models[k]:
            if k != "":
                key = f"{model.string(model)}{k}"
                comp_key = f"{model.string(model)}"
                df.loc[key, :] = all_shaps[key].loc[:, "atc"] - all_shaps[comp_key].loc[:, "atc"]

    return df

def plot_shap_move_by_zone(df):
    zones = df.columns
    versions = np.array([d.split("NN")[1] for d in df.index])
    versions_l = versions

    models = np.array([m.split(v)[0] for m, v in zip(df.index, versions)])
    models_l = models

    nm = len(np.unique(models))
    nv = int(len(models) / nm)
    umodels = models[range(0, len(versions), nv)]
    uversions = versions[range(0, nv)]
    
    xindices = np.array([d for d in range(len(zones))])

    params = {"fontsize" : 35, "fontsize_labels" : 23, "linewidth" : 3}    
    with matplotlib.rc_context({ "text.usetex" : True,
                                 "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc}",
                                 "font.family" : ""}):    
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        vabs = 0.5 * np.abs(df.values).max()
        im = ax.imshow(df.values, cmap="RdYlGn", vmin=-vabs, vmax=vabs);
     
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
            plt.annotate(model,
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

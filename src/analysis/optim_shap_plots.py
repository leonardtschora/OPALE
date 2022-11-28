from work.models.DNNWrapper import DNNWrapper
from work.models.Splitter import MySplitter
import os, pandas, numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as tck
from work.shapely_plots import normalize_shap
from work.flux_utils import *

def print_successess(diffs):
    k = "DNNFos"
    mi = map_model(k) - 1
    vi = map_version("\_"+k.split("NNF")[1]) - 1
    diff = diffs[mi][vi]
    
    ind_suc = (res.loc[k, :] == 1).values
    c_zones = zones[ind_suc]
    c_edges = []
    indices = []
    res_con = pandas.DataFrame()
    for i, cfrom in enumerate(c_zones):
        print(cfrom)
        for cto in list(zones) + model_wrapper.known_countries:
            edge = f"{cfrom}_{cto}"
            if edge in order[forder]:
                c_edges += [edge]
                indy = np.where(order[forder] == edge)[0][0]
                indices += [ind]
            print(edge, diff[i, indy])    

def fold_same_edge(df):
    already_done = []
    unique = []
    for c in df.columns:
        cfrom = c.split("_")[0]
        cto = c.split("_")[1]
        r = f"{cto}_{cfrom}"
        if (r not in already_done) and (c not in already_done):
            already_done += [r]
            already_done += [c]
            unique += [c]

            df.loc[:, c] += df.loc[:, r]
    return df.loc[:, unique]       

def plot_zone_cons(diffs, zone, res, index, order, forder, zones):
    zone = "NONO1"
    cs = list(order[forder][[o.split("_")[0] == zone for o in  order[forder]]])
    csback = [f"{c.split('_')[1]}_{zone}" for c in cs]
    edges = np.array(cs + csback)
    eindex = np.array([np.where(e == order)[0][0] for e in edges])
    zindex = np.where(zone == zones)[0][0]
    
    zdf = pandas.DataFrame(columns=edges, index=index)

    for i, k in enumerate(zdf.index):
        mi = map_model(k) - 1
        vi = map_version("\_"+k.split("NNF")[1]) - 1
        diff = diffs[mi][vi]
        zdf.loc[k, :] = diff[zindex, eindex]

    zdf = fold_same_edge(zdf)
    edges = zdf.columns
    vabs = np.abs(zdf.values).max()
    fig, ax = plt.subplots(1, figsize=(19.2, 10.8))     
    im = ax.imshow(
        zdf.values.astype(np.float64), vmin=-vabs, vmax=vabs, cmap="RdYlGn")
    ax.set_xticks(range(len(edges)))
    ax.set_xticklabels(edges, rotation=45)
    
    ax.set_yticks(range(len(index)))
    ax.set_yticklabels(index)

    #ys = np.array([d for d in np.where(res.loc[index, zone].values == 1)[0]])
    #for yss in ys:
    #    ax.axhline(yss, 0, 1, linestyle="--", c="g")
    
    #ys = np.array([d for d in np.where(res.loc[index, zone].values == -1)[0]])
    #for yss in ys:
    #    ax.axhline(yss, 0, 1, linestyle="--", c="r")    

    plt.colorbar(im, ax=ax, location="top", shrink=0.5)
    ax.set_title(f"{zone}")
    plt.show()

def df_diff(diff, threshold_plus, threshold_minus, zones, order, forder):
    xindices, yindices = np.where(
        np.logical_or(diff[:, forder] > threshold_plus,
                      diff[:, forder] < threshold_minus))
    res = pandas.DataFrame(
        columns=[
            "Label", "Feature", "ContribDiff"#, "link_from", "link_to"
        ],
        index = range(len(xindices)))
    res.Label = zones[xindices]
    res.Feature = order[forder][yindices]
    res.ContribDiff = diff[:, forder][xindices, yindices]

    """
    # show_link
    for i in res.index:
        z = res.loc[i, "Label"]
        zprime = res.loc[i, "Feature"].split("_")[0]
        forward = f"{z}_{zprime}"
        backward = f"{zprime}_{z}"
        if forward in order:
            res.loc[i, "link_from"] = diff[i, np.where(forward == order)[0][0]]
            res.loc[i, "link_to"] = diff[i, np.where(backward == order)[0][0]]
        else:
            res.loc[i, "link_from"] = 0.0
            res.loc[i, "link_to"] = 0.0
    """    
    return res

def load_and_normalize(model, version):
    folder = os.path.join(os.environ["VOLTAIRE"],"data", "datasets", "AllEuropeGNN")
    if version != "":
        version_ = f"_{version}"
    else: version_ = version
    filename = f"{model.string(model)}_TSCHORA{version_}_AllEuropeGNN_test_recalibrated_shape_values_0.npy"
    path = os.path.join(folder, filename)
    shaps = normalize_shap(np.load(path))
    return shaps

def get_dfs(model, version, results):
    shaps = load_and_normalize(model, version)
    model_wrapper = mw(model, version)

    key = f"{model.string(model)}{version}"
    results[key] = []

    results[key] += [by_zone(shaps.mean(axis=(0, 1)), model_wrapper)]
    results[key] += [by_zone_to_predict_zones(shaps.mean(axis=1), model_wrapper)]
    results[key] += [by_zone_to_predict_features(shaps.mean(axis=1), model_wrapper)]

def mw(model, version):
    model_wrapper = model(
        f"{model.string(model)}_TSCHORA", "AllEuropeGNN",
        replace_ATC = version, spliter = MySplitter(365, shuffle=False),
        known_countries = ["CH", "GB"], countries_to_predict = "all")
    X, Y = model_wrapper.load_train_dataset()
    return model_wrapper

def sum_hours(values):
    values = values[:, 1:]
    ox, oy = int(values.shape[0]/24), int(values.shape[1]/24)
    res = np.zeros((ox, oy))
    for x in range(ox):
        for y in range(oy):
            res[x, y] = values[24*x:24*(x+1), 24*y:24*(y+1)].sum()
    return res / 24       
        
def by_day(values, model_wrapper, func):
    res = []
    for i in range(values.shape[0]):
        res += [func(values[i, :], model_wrapper)]
    return res

def by_zone(values, model_wrapper, normalize=False, connections=True):
    zones = model_wrapper.countries_to_predict + model_wrapper.known_countries
    features = ["consumption", "production", "renewables_production", "price"]
    if connections:
        features += ["atc_from", "atc_to"]
    
    results = pandas.DataFrame(index=zones, columns=features,
                               data=np.zeros((len(zones), len(features))))
    for z in zones:
        for (i, f) in enumerate(features):
            if f in ("atc_from", "atc_to"):
                if f == "atc_from":
                    keys = [f"{z}_{zprime}" for zprime in zones]
                elif f == "atc_to":
                    keys = [f"{zprime}_{z}" for zprime in zones]
                count_con = 0
                for k in keys:
                   inds = model_wrapper.edges_columns_idx[k]
                   if len(inds) > 0:
                       results.loc[z, f] += np.sum(values[inds])
                       count_con += 1
                       
                divider = 1
                if normalize: divider = 24 * max(count_con, 1)
                results.loc[z, f] /= 2 * divider
            else:
                divider = 1
                if normalize: divider = 24
                results.loc[z, f] = np.sum(
                    values[model_wrapper.country_idx[z][24*i:24*(i+1)]]) / divider

    return results

def by_zone_to_predict_zones(values, model_wrapper,
                             connections=True, normalize=False):
    zones_label = model_wrapper.countries_to_predict
    zones = model_wrapper.countries_to_predict + model_wrapper.known_countries
    
    results = pandas.DataFrame(index=zones_label, columns=zones,
                               data=np.zeros((len(zones_label), len(zones))))
    for i, z in enumerate(zones_label):
        for zf in zones:
            xindices = np.array(
                [i for i in range(i * 24, (i + 1) * 24)]).reshape(-1, 1)
            yindices = np.array(model_wrapper.country_idx[zf])
            
            d = 1
            if normalize: d = len(yindices)
            results.loc[z, zf] = np.sum(values[xindices, yindices])/d

            if connections:
                # Add the edges
                count_con_1 = 0
                to_add_1 = 0
                count_con_2 = 0
                to_add_2 = 0            
                for zff in zones:
                    k1 = f"{zf}_{zff}"
                    inds = model_wrapper.edges_columns_idx[k1]
                    if len(inds) > 0:
                        to_add_1 += np.sum(values[xindices, inds])
                        count_con_1 += 1
                        
                    k2 = f"{zff}_{zf}"
                    inds = model_wrapper.edges_columns_idx[k2]
                    if len(inds) > 0:
                        to_add_2 += np.sum(values[xindices, inds])
                        count_con_2 += 1
                        
                if count_con_1 > 0 and count_con_2 > 0:
                    d1 = 2
                    d2 = 2
                    if normalize:
                        d1 = 2 * count_con_1
                        d2 = 2 * count_con_2
                    results.loc[z, zf] += (to_add_1 / d1 + to_add_2 / d2)
        results.loc[z] = 100 * results.loc[z] / results.loc[z].sum()

    return results

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
    return results

def plot_most_contributive_countries(df):
    by_countries = df.sum(axis=1)
    by_countries.sort_values(ascending=False, inplace=True)

    width = 0.75
    xindices = range(len(by_countries))
    
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    ax.bar(xindices, by_countries.values, width=width)

    ax.set_xticks(xindices)
    ax.set_xticklabels(by_countries.index, rotation=45)

    ax.yaxis.set_minor_locator(tck.MultipleLocator(0.1))

    ax.set_xlabel("Zones")
    ax.set_ylabel("Contribution \%")
    ax.grid("on", which="major", linestyle="-", linewidth=1, color="k")
    ax.grid("on", which="minor", linestyle="--", linewidth=0.5, color="k")


    ax.set_xlim([0 - 0.5, len(by_countries) - width / 2])
    ax.set_title("Contribution of each zone in the forecasting task")
    
    plt.show()

def plot_zones_for_zones(df):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    im = ax.imshow(df.values, vmin=0, cmap="Reds")

    fig.colorbar(im)

    ax.set_xlabel("Features contributions") 
    ax.set_ylabel("Prdicted zones")

    nz, nzz = df.values.shape
    ax.set_xticks(range(nzz))
    ax.set_xticklabels(df.columns, rotation=45)    
    ax.set_yticks(range(nz))
    ax.set_yticklabels(df.index)

    plt.show()
    
def plot_features_for_zones(df, stacked=False):
    fig, ax = plt.subplots(figsize=(19.2, 10.8))

    total_width = 0.8
    bar_width = total_width / len(df.columns)
    colors = ["c", "r", "b", "m"]
    cmap = plt.get_cmap("viridis")
    previous = np.zeros(len(df.index))
    for i, feature in enumerate(df.columns):
        if not stacked: bar_width_ = bar_wdith
        else: bar_width_ = 0
        xindices = [j + i * bar_width_ for j in range(len(df.index))]
        ci = i / (len(df.columns) - 1)

        if not stacked: bar_width_ = bar_wdith
        else: bar_width_ = total_width
        
        ax.bar(xindices, df.loc[:, feature], bottom=previous,
               color=cmap(ci), width=bar_width_, label=feature)
        if stacked:
            previous += df.loc[:, feature]

    ax.grid("on", axis="y", which="major", linestyle="-", linewidth=1, color="k")
    ax.grid("on", which="minor", linestyle="--", linewidth=0.5, color="k")
    ax.set_xlabel("Zones to forecast")
    ax.set_ylabel("Contribution \%")

    if not stacked:
        ax.set_ylim([df.min().min(), df.max().max()])
    else:
        ax.set_ylim([0, 100])
    ax.set_xlim([0 - 0.5, len(df.index) - total_width / 2])
    
    ax.set_xticks([j for j in range(len(df.index))])
    ax.set_xticklabels(df.index, rotation=45)
    ax.yaxis.set_minor_locator(tck.MultipleLocator(1))

    ax.set_title("Contribution for each zone, summed by features")
    plt.legend()
    plt.show()
        
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
        
def get_diffs(model, versions, perc=False):
    comp = load_and_normalize(model, "")
    shour = sum_hours(comp.mean(axis=1))
    va = []
    diffs=[]
    for i, m in enumerate(versions):
        values = load_and_normalize(model, m) # space = label % 
        vmean = values.mean(axis=1) # space = label % 
        vhour = sum_hours(vmean) # space = label %
        n = vhour - shour # space = diff of %
        if perc:
            d = np.where(shour == 0, 0.0001, shour)
            diff = 100 * n / d
        else:
            diff = n
        diffs += [diff]
    return diffs

def get_hours(model, versions, perc=False):
    va = []
    diffs=[]
    for i, m in enumerate(versions):
        values = load_and_normalize(model, m) # space = label % 
        vmean = values.mean(axis=1) # space = label % 
        vhour = sum_hours(vmean) # space = label %
        diffs += [vhour]
    return diffs

def plot_contribdiff_links(models, order, forder):
    versions = ["WithPrice", "LeastSquares", "combined", "combined_unilateral"] 
    df = pandas.DataFrame(columns=[map_model_name(m.string(m))+ map_Flux_versions(v)
                                   for m in models
                                   for v in versions],
                          index=order[forder])
    for m in models:
        diffs = get_diffs(m, versions)
        for i, v in enumerate(versions):
            col = map_model_name(m.string(m))+ map_Flux_versions(v)
            df.loc[:, col] = diffs[i].mean(axis=0)[forder]


    df = df.transpose()
    already_done = []
    uniques = []
    for c in df.columns:
        cfrom = c.split("_")[0]
        cto = c.split("_")[1]
        opposite_col = f"{cto}_{cfrom}"
        if (c not in already_done) and (opposite_col not in already_done):
            uniques += [c]
            df.loc[:, c] = df.loc[:, opposite_col]
            already_done += [c]
            already_done += [opposite_col]    
    df = df.loc[:, uniques]    
    df = df.transpose().sort_index().transpose()        
    return df
    

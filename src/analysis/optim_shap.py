import numpy as np, os
import matplotlib.pyplot as plt
from work.shapely_plots import normalize_shap
from work.analysis.metrics_utils import df_to_latex
from work.models.CNNWrapper import CNNDropWrapper
from work.models.DNNWrapper import DNNWrapper
from work.models.RFR import RFR
from work.models.GNNWrapper import NodeGNNWrapper

%load_ext autoreload
%autoreload 1
%aimport work.optim_shap_plots

from work.optim_shap_plots import *
##########
results = {}
models = [DNNWrapper, CNNDropWrapper, NodeGNNWrapper]
versions = ["", "WithPrice", "LeastSquares", "combined", "combined_unilateral"]
for model in models:
    for version in versions:
        get_dfs(model, version, results)

df = pandas.DataFrame(columns=results["DNN"][2].columns, index=list(results.keys()),
                      dtype=np.float64)
for model in models:
    for version in versions:
        key = f"{model.string(model)}{version}"
        df.loc[key, :] = results[key][2].mean()

df.index = [map_model_name(d)+"\_"+map_Flux_versions(d.split("NN")[1])
            for d in df.index]
df.columns = ["C", "G", "R", "P", "F"]
dividers = np.array([35, 35, 35, 35, 126])
df  = (df / dividers)
for i in df.index:
    df.loc[i] = 100 * df.loc[i] / df.loc[i].sum()

print(df_to_latex(df, roundings=1, hlines=False))
plot_features_summary(df)

results_table = copy.deepcopy(results)
df_table = copy.deepcopy(df)
########## Zone by zones
results = copy.deepcopy(results_table)
df = pandas.DataFrame(index=[d for d in results.keys() if len(d) > 7],
                      columns=results["DNN"][2].index,
                      dtype=np.float64)
for model in models:
    for version in versions:
        if version != "":
            key = f"{model.string(model)}{version}"
            comp_key = f"{model.string(model)}"
            df.loc[key, :] = results[key][2].loc[:, "atc"] - results[comp_key][2].loc[:, "atc"]

with matplotlib.rc_context({ "text.usetex" : True,
                             "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc}",
                             "font.family" : ""}):
    params = {"fontsize" : 35, "fontsize_labels" : 23, "linewidth" : 3}
    plot_shap_move_by_zone(df, params, res)

df_plot = copy.deepcopy(df)
########## Individual model
order = model_wrapper.columns[range(0, len(model_wrapper.columns), 24)]
forder = np.array([c.split("_")[0] == "ATC" for c in  order])
order = np.array([map_fname(o) for o in order])

df = plot_contribdiff_links(models, order, forder)
df_comp = copy.deepcopy(df)

res = plot_DM_results_by_zone(model_wrappers, zones, pvalues_smape, params)
mss = [map_model_name(m.string(m))for m in models]
res.index = [mss[i // len(versions)] + v for i, v in enumerate(res.index)]
index = res.index

diffs = [get_diffs(model, versions, perc=False) for model in models]

top_k = df.loc[k, :].sort_values().tail(5)
linked_zones = np.unique(np.concatenate([i.split("_") for i in top_k.index]))
linked_zones = [latex_zone(z) for z in linked_zones if z not in ("CH", "GB")]
print(res.loc[k, linked_zones], top_k)

fig, axes = plt.subplots(2, 1, figsize=(19.2, 10.8))
for i in range(2):
    ax = axes[i]
    cols = df.columns[32*i:32*(i+1)]
    ax.imshow(df.loc[:, cols],
              cmap="seismic", vmin=-25, vmax=25)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
plt.show()

df_to_print = df.loc[:, df.mean().sort_values().tail(10).sort_values(ascending=False).index].round(decimals=3)

print(df_to_latex(df_to_print, hlines=False))

dhour = [get_hours(model, versions, perc=False) for model in models]
ds = np.concatenate([d for d in dhour])
df_ds = pandas.DataFrame(columns=order[forder], index=res.index,
                         data=ds[:, :, forder].mean(axis=1))
df_ds = fold_same_edge(df_ds)

fig, ax = plt.subplots(1, figsize=(19.2, 10.8))
vabs = 1
mmin = 0.5
ax.imshow(np.where(df_ds.values == 0, mmin, df_ds.values), vmin=mmin,
          vmax=0.5*df_ds.values.max(), cmap="Greens")

ax.set_xticks(range(len(df_ds.columns)))
ax.set_xticklabels(df_ds.columns, rotation=45)

ax.set_yticks(range(len(df_ds.index)))
ax.set_yticklabels(df_ds.index)
plt.show()


version = "combined_unilateral"
v = map_Flux_versions(version)
vi = map_version("\_"+v[1:]) - 1
mi = 0
m = "DNN"

threshold_plus = 0.0
pthreshold_minus = 0.0
df = df_diff(diffs[mi][vi], threshold_plus, threshold_minus, zones, order, forder)
#df = df.loc[[l in f for l, f in zip(df.Label, df.Feature)]]

df.loc[:, "LabelLatex"] = [latex_zone(z) for z in df.Label]
df.loc[:, "improvements"] = res.loc[m+v, df.LabelLatex].values
col_metr = f"GNN\_{v}"
col_ref = f"GNN\_A"
df.sort_values("ContribDiff", inplace=True)

vabs = 0.5
fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
axes = axes.flatten()
for i, m in enumerate(versions):
    ax = axes[i]
    ax.set_title(m)
    diff = diffs[mi][i]
    im = ax.imshow(diff, vmin=-vabs, vmax=vabs, cmap="seismic")

    improved = np.unique(
        np.concatenate(
            (np.where(diff > vabs)[1],             
             np.where(diff.mean(axis=0) > vabs/5)[0],
             np.where(diff.mean(axis=0) < -vabs/5)[0])))
    ax.set_xticks(improved)
    ax.set_xticklabels(order[improved], rotation=45)

    yindices = range(len(zones))
    ax.set_yticks(yindices)
    ax.set_yticklabels(zones)

plt.colorbar(im, ax=axes, location="top", shrink=0.6)
plt.show()
##########
folder = os.path.join(os.environ["VOLTAIRE"], "data", "datasets", "AllEuropeGNN")
path_1 = os.path.join(folder, "DNN_TSCHORA_AllEuropeGNN_test_recalibrated_shape_valu
es.npy")
path_2 = os.path.join(folder, "DNN_TSCHORA_AllEuropeGNN_test_recalibrated_shape_values_0.npy")

m1 = np.load(path_1)
m2 = np.load(path_2)

m1.shape == m2.shape
m1.dtype
m2.dtype

# Some errors occur from the type conversion
diff_64 = m1 - m2
diff_16 = diff_64.astype(np.float16)
[np.mean(diff_64), np.max(diff_64), np.min(diff_64)]
[np.mean(diff_16), np.max(diff_16), np.min(diff_16)]

# Values are well-centered
values, bins = np.histogram(diff_64, bins=100)
# but their sum is significant
np.abs(diff_64).sum()

# SO let's keep m2 instead of m1!
values, bins = np.histogram(m2, bins=100)
width = np.mean(bins[1:] - bins[:-1])
plt.bar(bins[1:], values, width=width)
plt.show()

# Contribution sums to 1 for every day
mn = normalize_shap(m2)
mn = np.abs(m2)

# Check mean across dims
model_wrapper = mw()

# keep features
mean_features = mn.mean(axis=(0, 1))
df = by_zone(mean_features, model_wrapper, connections=False)
df = by_zone(mean_features, model_wrapper, normalize=True) # Unit contribution
df = by_zone(mean_features, model_wrapper)
df.mean(axis=1), df.mean() # Nothing to see here....

# Plot most contributive countries
plot_most_contributive_countries(df)

# Same thing but evergy day
mean_days_features = mn.mean(axis=0)
dfs = by_day(mean_days_features, model_wrapper, by_zone)

# mean (days)
mean_labels_features = mn.mean(axis=1)

# Plot the contribution of each zone
df = by_zone_to_predict_zones(mean_labels_features, model_wrapper,
                              connections=False, normalize=False)
plot_zones_for_zones(df)

# Plot the contribution of each feature
df = by_zone_to_predict_features(mean_labels_features, model_wrapper,
                                 connections=True, normalize=False)
plot_features_for_zones(df, stacked=True)



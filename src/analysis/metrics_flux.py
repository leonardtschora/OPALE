import pandas, matplotlib.pyplot as plt, numpy as np, datetime, os, copy
from work.optimize_flux_utils import load_joined
from work.models.GNNWrapper import GNNWrapper
from work.analysis.evaluate import ACC
from work.analysis.metrics_utils import df_to_latex

# Load Data
model_wrapper = GNNWrapper("FLUX", "AllEuropeGNN", "FR", known_countries=[])
X, Y = model_wrapper.load_train_dataset()
ptemp = model_wrapper.params()
ptemp["n_epochs"] = 1

regr = model_wrapper.make(ptemp)
regr.fit(X, Y)
model = regr.regressor_.steps[1][1]
dates, Xe, Xn = model.split_node_edge(X)
datesh = [datetime.datetime(d.year, d.month, d.day, h)
          for d in dates for h in range(24)]
power_max = pandas.read_csv(os.path.join(
    os.environ["VOLTAIRE"], "data", "datasets", "PowerMax","joined.csv"))
power_max.index = [datetime.datetime.strptime(
    d, "%Y-%m-%dT%H:00:00.0") for d in power_max.period_start_time]
power_max = power_max.loc[datesh]
connections = pandas.read_csv(model_wrapper.connection_path, index_col="Country")

# Some constants
n = Xn.shape[0] * 24
nz = len(model.country_list)

# Compute the number of connections
all_edges = list(model.edges_columns_idx.keys())
all_con = np.array([e for e in all_edges if len(model.edges_columns_idx[e]) > 0])
all_out = np.array([c.split("_")[0] for c in all_con])
all_in = np.array([c.split("_")[-1] for c in all_con])

T = len(all_con)

######################################## PROD ANALYSIS
# LOAD Production
run = "WithPrice"
real_prod, real_prod_reshaped = load_joined("RealProd", "real_production", dates,
                                            n, nz, model.country_list,suffix="Prog")
base_path = os.path.join(os.environ["VOLTAIRE"], "data", "datasets")
df_prod_hourly = pandas.read_csv(
    os.path.join(base_path, "OnlyProdGNN", f"joined_{run}_hourly.csv"),
    index_col="Unnamed: 0")
df_prod = pandas.read_csv(
    os.path.join(base_path, "OnlyProdGNN", f"joined_{run}.csv"),
    index_col="Unnamed: 0")

glob_acc = round(ACC(real_prod_reshaped, df_prod_hourly.values), ndigits=3)
print(f"Global ACC for {run} is {glob_acc}")

accs = ACC(real_prod_reshaped, df_prod_hourly.values, mean=False)
maes = np.abs(real_prod_reshaped - df_prod_hourly.values).mean(axis=0)
smapes = 200 * (np.abs(real_prod_reshaped - df_prod_hourly.values) / (np.abs(real_prod_reshaped) + np.abs(df_prod_hourly.values))).mean(axis=0)
metrics = [accs, maes, smapes]
ms = ["accs", "maes", "smapes"]

xindices = range(nz)
fig, axes = plt.subplots(3, 1, figsize=(19.2, 10.8),
                         sharex=True, gridspec_kw={"hspace" : 0, "wspace" : 0})
for i, m in enumerate(ms):
    ax = axes[i]
    ax.bar(xindices, metrics[i])
    ax.grid("on")
    ax.set_title(m, y=0.8)
ax.set_xticks(xindices)
ax.set_xticklabels(model.country_list, rotation=45)
plt.suptitle("Metrics of programmable production forecast")
plt.show()

# CHECK SUM
# Some Plots : conso, prod and renewables
datesh = [datetime.datetime(d.year, d.month, d.day) + datetime.timedelta(hours=h) for d in dates for h in range(24)]
fig, axes = plt.subplots(7, 5, figsize=(19.2, 10.8),
                         sharex=True,
                         sharey=True,
                         gridspec_kw={"hspace" : 0, "wspace" : 0})
axes = axes.flatten()
for i, (ax, c) in enumerate(zip(axes, model.country_list)):
    ax.plot(datesh, real_prod_reshaped[:, i], c="r", label="REAL PROD")
    ax.plot(datesh, df_prod_hourly.values[:, i], c="c", label="PROGRAMMABLE PROD")
    
    ax.set_title(f"{c}", y=0.75)
    ax.set_xticks(range(0, len(dates), 24*100))
    ax.grid("on")
plt.legend()
plt.show()

results = pandas.DataFrame(index=model.country_list, columns=ms,
                           data=np.array(metrics).transpose())
print(df_to_latex(results, hlines=False))

######################################## FLUX ANALYSIS
# LOAD Realized FLux
run = "NoPrice"
real_flux = pandas.read_csv(os.path.join(
    os.environ["VOLTAIRE"], "data", "datasets", "Flux", "joined.csv"))
real_flux.index = [datetime.datetime.strptime(
    d, "%Y-%m-%d") for d in real_flux.period_start_date]
real_flux = real_flux.loc[dates]
real_flux.drop(columns="period_start_date", inplace=True)

all_edges = list(model.converted_edge_indices.keys())
real_flux = real_flux.loc[:, ["Flux" + c[3:] for c in model_wrapper.edges_columns]]
real_flux = real_flux.values
real_flux_reshaped = np.zeros((n, T))
for i in range(T):
    real_flux_reshaped[:, i] = real_flux[:, 24 * i : 24 * (i + 1)].reshape(-1)


base_path = os.path.join(os.environ["VOLTAIRE"], "data", "datasets")
df_prod_hourly = pandas.read_csv(
    os.path.join(base_path, "OnlyProdGNN", f"joined_{run}_hourly.csv"),
    index_col="Unnamed: 0")
df_prod = pandas.read_csv(
    os.path.join(base_path, "OnlyProdGNN", f"joined_{run}.csv"),
    index_col="Unnamed: 0")

glob_acc = round(ACC(real_prod_reshaped, df_prod_hourly.values), ndigits=3)
print(f"Global ACC for {run} is {glob_acc}")

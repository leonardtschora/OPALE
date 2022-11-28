import numpy as np, pandas, copy, itertools, math, copy, datetime
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from work.analysis.evaluate import mae, smape, mape, rmse, rmae, dae, cmap, cmap_2
from work.analysis.evaluate import cmap_scaled_values, cmap_diff_values, cmap_diff_values_2
from work.analysis.evaluate import load_prevs_mw
from work.models.LAGO_wrapper import LAGOWrapper
from work.models.CNNWrapper import CNNWrapper
from work.models.Feature import Naive
from work.models.Splitter import MySplitter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from work.columns import labels_and_pos
import work.TSCHORA_results_utils as tru
import matplotlib.ticker as tck
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy import stats

class MinAbsScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self, epsilon=10e-5):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        min_abs = np.abs(np.array(X)).min()
        min_abs = np.clip(min_abs, a_min=self.epsilon, a_max=None)
        self.min_abs = min_abs.reshape(-1, 1)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = np.array(X) / self.min_abs
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = np.array(X) * self.min_abs
        return transformed_data
    
def plot_error_correlation(country, version, dataset, model_wrapper_index,
                           naive_forecasts, real_prices, predictions,
                           model_wrappers):
    model_wrapper, ypred, ytrue, ynaive, dates = get_all(
        country, version, dataset, model_wrapper_index,
        naive_forecasts, real_prices, predictions, model_wrappers)
    
    test_set = pandas.read_csv(model_wrapper.test_dataset_path())
    test_set.drop(columns=model_wrapper.label + ["period_start_date"], inplace=True)
    feature_names_ordered, feature_labels, label_pos, sort_indices, lags = labels_and_pos(model_wrapper, version, country, correl=True)
    data = test_set.values[:, sort_indices]
    
    ytrue_daily = ytrue.mean(axis=1)
    ypred_daily = ypred.mean(axis=1)
    mae = np.abs(ytrue_daily - ypred_daily)
    
    temp = np.zeros((data.shape[0], data.shape[1] + 1))
    temp[:, :-1] = data
    temp[:, -1] = mae
    corrs = np.corrcoef(temp.transpose())[:-1, -1]
    idxs = range(len(corrs))
    
    plt.scatter(idxs, corrs)
    plt.xticks([l - 1/2 for l in label_pos], feature_labels, rotation=45)
    plt.grid("on", which="both", axis="both")
    plt.ylabel("Correlation")
    plt.title("Correlation with the MAE")

    
def plot_feature_correlation(country, version, dataset, model_wrapper_index,
                             naive_forecasts, real_prices, predictions,
                             model_wrappers):
    model_wrapper = model_wrappers[version][dataset][country][model_wrapper_index]
    test_set = pandas.read_csv(model_wrapper.test_dataset_path())
    test_set.drop(columns=model_wrapper.label + ["period_start_date"], inplace=True)
    feature_names_ordered, feature_labels, label_pos, sort_indices, lags = labels_and_pos(model_wrapper, version, country, correl=True)
    data = test_set.values[:, sort_indices]
    im = plt.imshow(np.corrcoef(data.transpose()), cmap="seismic",
                    vmin=-1, vmax=1)
    plt.xticks([l - 1 for l in label_pos], feature_labels, rotation=45)
    plt.yticks([l - 1 for l in label_pos], feature_labels)
    plt.colorbar(im)   
    
def plot_error_repartition(country, version, dataset, model_wrapper_index,
                           naive_forecasts, real_prices, predictions,
                           model_wrappers, daily=False, n_bins=100):
    """
    ENABLE MULTIPLE PLOTS
    """
    model_wrapper, ypred, ytrue, ynaive, dates = get_all(
        country, version, dataset, model_wrapper_index,
        naive_forecasts, real_prices, predictions, model_wrappers)
    
    daily = daily or (len(model_wrapper.label) == 1)
    if daily:
        ytrue_daily = ytrue.mean(axis=1)
        ypred_daily = ypred.mean(axis=1)
        smape = 200 * np.abs(ytrue_daily-ypred_daily) / (np.abs(ytrue_daily) + np.abs(ypred_daily))
        mae = np.abs(ytrue_daily - ypred_daily)
    else:
        smape = 200 * np.abs(ytrue-ypred) / (np.abs(ytrue) + np.abs(ypred))
        mae = np.abs(ytrue - ypred)
        smape = smape.reshape(-1, 1)
        mae = mae.reshape(-1, 1)    

    fig, axsmape = plt.subplots(figsize=(19.2, 10.8))
    axmae = axsmape.twinx()
    axmae.set_ylabel("MAE \euro /MWh")
    axsmape.set_ylabel("SMAPE \%")    
    for ax, data, l, c in zip([axsmape, axmae], [smape, mae], ["smape", "mae"],
                              ["b", "r"]):
        start = data.min()
        stop = np.quantile(data, 0.95)
        bins = np.linspace(start, stop, n_bins)
        ys, xs = np.histogram(data, bins=bins)        
        xs = (xs[1:] + xs[:-1])/2
        ys = 100 * np.cumsum(ys) / ys.sum()
        ax.plot(ys, xs, alpha=1, c=c, linewidth=3, label=l)
        ax.legend()
        ax.grid("on", which="both", axis="both")        
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=10))

    plt.title(f"Error repartition for {model_wrapper.prefix} on {model_wrapper.dataset_name}")

def compute_moments(country, versions, datasets, model_wrapper_index,
                    naive_forecasts, real_prices, predictions, model_wrappers,
                    daily=False):
    dfs = {"mae" : pandas.DataFrame(columns=["Mean", "Std", "0.5", "0.75", "0.9"]),
           "smape" :pandas.DataFrame(columns=["Mean", "Std", "0.5", "0.75", "0.9"])}
    for version in versions:
        for dataset in datasets:
            model_wrapper, ypred, ytrue, ynaive, dates = get_all(
                country, version, dataset, model_wrapper_index,
                naive_forecasts, real_prices, predictions, model_wrappers)

            daily = daily or (len(model_wrapper.label) == 1)
            if daily:
                ytrue_daily = ytrue.mean(axis=1)
                ypred_daily = ypred.mean(axis=1)
                smape = 200 * np.abs(ytrue_daily-ypred_daily) / (np.abs(ytrue_daily) + np.abs(ypred_daily))
                mae = np.abs(ytrue_daily - ypred_daily)
            else:
                smape = 200 * np.abs(ytrue-ypred) / (np.abs(ytrue) + np.abs(ypred))
                mae = np.abs(ytrue - ypred)
                smape = smape.reshape(-1, 1)
                mae = mae.reshape(-1, 1) 

            for data, l in zip([smape, mae], ["smape", "mae"]):
                idx = f"{version}_{dataset}"
                avg = round(data.mean(), ndigits=2)
                std = round(data.std(), ndigits=2)
                md = round(np.quantile(data, 0.5), ndigits=2)
                qt1 = round(np.quantile(data, 0.75), ndigits=2)
                qt2 = round(np.quantile(data, 0.9), ndigits=2)
                dfs[l].loc[idx, :] = [avg, std, md, qt1, qt2]
                
    return dfs                       
    
def get_all(country, dataset, s, model_wrapper_index,
            naive_forecasts, real_prices, predictions, model_wrappers):
    """
    Return the model wrapper, predictions, true labels, naive forecasts and
    the dates of the specified country, dataset, set and model_wrapper  index.
    """
    keys = [k for k in predictions[dataset][s][country].keys()]
    model_wrapper = keys[model_wrapper_index]
    mw = [k for k in model_wrappers[dataset][s][country]][model_wrapper_index]
    ypred = predictions[dataset][s][country][mw]
    ytrue = real_prices[s][country]
    ynaive = naive_forecasts[s][country]
    dates = [datetime.datetime.strptime(d, "%Y-%m-%d")
             for d in pandas.read_csv(model_wrapper.test_dataset_path()
             ).period_start_date]

    return model_wrapper, ypred, ytrue, ynaive, dates


def plot_error_byhour(country, version, dataset, model_wrapper_index,
                      naive_forecasts, real_prices, predictions, model_wrappers):
    factor = 1
    fig, axes = plt.subplots(1, 3, figsize=(factor * 19.2, factor * 10.8))
    model_wrapper, ypred, ytrue, ynaive, dates = get_all(
        country, version, dataset, model_wrapper_index,
        naive_forecasts, real_prices, predictions, model_wrappers)    

    smape = 200 * np.abs(ytrue-ypred) / (np.abs(ytrue) + np.abs(ypred))
    mae = np.abs(ytrue - ypred)
    signed_mae = ytrue - ypred

    smape_avg = smape.mean(axis=0)
    mae_avg = mae.mean(axis=0)
    signed_mae_avg = signed_mae.mean(axis=0)

    xindices = range(24)    
    for i, (metric, metric_str) in enumerate(
            zip([mae_avg, signed_mae_avg, smape_avg],
                ["mae", "signed_mae", "smape"])):
        ax = axes[i]
        ax.bar(xindices, metric)

        if metric.min() < 0: ymin = 1.1 * metric.min()
        else: ymin = 0.8 * metric.min()
        ymax = 1.05 * metric.max()

        ax.grid("on")
        ax.set_ylim([ymin, ymax])
        avg = metric.mean()
        ax.plot([i - 0.5 for i in range(25)],
                [avg for i in range(25)], lw=3, ls="--", c="r", label="Mean")
        offpeak = np.concatenate((metric[:8], metric[-4:])).mean()
        ax.plot([i - 0.5 for i in range(25)],
                [offpeak for i in range(25)], lw=3, ls="--", c="b", label="offpeak")
        peak = metric[8:-4].mean()
        ax.plot([i - 0.5 for i in range(25)],
                [peak for i in range(25)], lw=3, ls="--", c="g", label="PeakLoad")
        ax.set_xlabel("Hour")
        ax.set_title(metric_str)        
        
    ax.legend()
    plt.show()

def plot_error_byweekday(country, version, dataset, model_wrapper_index,
                         naive_forecasts, real_prices, predictions, model_wrappers):
    factor = 1
    fig, axes = plt.subplots(1, 3, figsize=(factor * 19.2, factor * 10.8))
    model_wrapper, ypred, ytrue, ynaive, dates = get_all(
        country, version, dataset, model_wrapper_index,
        naive_forecasts, real_prices, predictions, model_wrappers)    

    smape = 200 * np.abs(ytrue-ypred) / (np.abs(ytrue) + np.abs(ypred))
    mae = np.abs(ytrue - ypred)
    signed_mae = ytrue - ypred
    if "BL" in version:
        smape_daily = smape.reshape(-1, 1)
        mae_daily = mae.reshape(-1, 1)
        signed_mae_daily = signed_mae.reshape(-1, 1)
    else:
        ytrue_daily = ytrue.mean(axis=1)
        ypred_daily = ypred.mean(axis=1)                
        smape_daily = 200 * np.abs(ytrue_daily-ypred_daily) / (np.abs(ytrue_daily) + np.abs(ypred_daily))
        mae_daily = np.abs(ytrue_daily - ypred_daily)
        signed_mae_daily = ytrue_daily - ypred_daily

    weekdays = np.array([d.weekday() for d in dates])
    metrics = np.zeros((3, 7))
    ms = [mae_daily, signed_mae_daily, smape_daily]
    for d in range(7):
        indices = np.where(weekdays == d)[0]
        for i in range(3):
            metrics[i, d] = np.mean(ms[i][indices])

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, (ax, metric_str) in enumerate(zip(axes, ["mae", "signed_mae", "smape"])):
        ax.bar(range(7), metrics[i, :])
        ax.set_xticks(range(7))
        ax.set_xticklabels(day_names, rotation=45)

        if metrics[i, :].min() < 0: ymin = 1.1 * metrics[i, :].min()
        else: ymin = 0.8 * metrics[i, :].min()
        ymax = 1.05 * metrics[i, :].max()

        ax.grid("on")
        ax.set_ylim([ymin, ymax])
        avg = metrics[i, :].mean()
        ax.plot([i - 0.5 for i in range(8)],
                [avg for i in range(8)], lw=3, ls="--", c="r")
        ax.set_xlabel("WeekDay")
        ax.set_title(metric_str)

    plt.show()

def plot_error_overtime(country, datasets, sets, model_wrapper_indices,
                        naive_forecasts, real_prices, predictions,
                        model_wrappers, metric="mae"):
    factor = 1
    fig, ax = plt.subplots(1, figsize=(factor * 19.2, factor * 10.8))
    colors = ["b", "r"]
    for k, model_wrapper_index in enumerate(model_wrapper_indices):
        for  i, s in enumerate(sets):
            for j, dataset in enumerate(datasets):
                model_wrapper, ypred, ytrue, ynaive, dates = get_all(
                    country, dataset, s, model_wrapper_index,
                    naive_forecasts, real_prices, predictions, model_wrappers)
            
                smape = 200 * np.abs(ytrue-ypred) / (np.abs(ytrue) + np.abs(ypred))
                mae = np.abs(ytrue - ypred)
                signed_mae = ytrue - ypred
                
                if "BL" in dataset:
                    smape_daily = smape.reshape(-1, 1)
                    mae_daily = mae.reshape(-1, 1)
                    signed_mae_daily = signed_mae.reshape(-1, 1)
                    mapping = {"smape" : smape_daily,
                               "mae" : mae_daily,
                               "signed_mae" : signed_mae_daily}
                    ymin = mapping[metric].min()
                    ymax = mapping[metric].max()
                    ax.plot(dates, mapping[metric],
                            label=f"{model_wrapper.string()}_{dataset}_{s}",
                            alpha=0.7)                
                else:
                    smape_hourly = smape.reshape(-1, 1)
                    mae_hourly = mae.reshape(-1, 1)
                    signed_mae_hourly = signed_mae.reshape(-1, 1)
                    
                    ytrue_daily = ytrue.mean(axis=1)
                    ypred_daily = ypred.mean(axis=1)                
                    smape_daily = 200 * np.abs(ytrue_daily-ypred_daily) / (np.abs(ytrue_daily) + np.abs(ypred_daily))
                    mae_daily = np.abs(ytrue_daily - ypred_daily)
                    signed_mae_daily = ytrue_daily - ypred_daily
                    
                    dates_hourly = np.concatenate([np.array(
                        [d + datetime.timedelta(hours=h) for h in range(24)])
                                                   for d in dates])
                    mapping = {"smape" : [smape_hourly, smape_daily],
                               "mae" : [mae_hourly, mae_daily],
                               "signed_mae" : [signed_mae_hourly, signed_mae_daily]}
                    ymin = mapping[metric][0].min()
                    ymax = mapping[metric][0].max()                
                    ax.plot(dates_hourly, mapping[metric][0],
                            label=f"{model_wrapper.string()}_{dataset}_{s}",
                            alpha=0.3)
                    ax.plot(dates, mapping[metric][1],
                            label=f"{model_wrapper.string()}_{dataset}_{s}",
                            alpha=0.7)                
                
    # Display lockdown dates
    dates_to_plot = [datetime.datetime(2020, 3, 17),
                     datetime.datetime(2020, 5, 11),
                     datetime.datetime(2020, 10, 30),
                     datetime.datetime(2020, 12, 15),
                     datetime.datetime(2021, 4, 3),
                     datetime.datetime(2021, 5, 3)]

    for d in dates_to_plot:
        ax.plot([d, d], [ymin, ymax], c="k", linestyle="--", linewidth=2)

    patches = []
    for i in range(3):
        d1 = mdates.date2num(dates_to_plot[2 * i])
        d2 = mdates.date2num(dates_to_plot[2 * i + 1])
        patches.append(mpatches.Rectangle((d1, ymin), d2 - d1, ymax - ymin))

    collection = PatchCollection(patches, alpha=0.3, facecolor="grey")
    ax.add_collection(collection)

    # Pretty axes
    delta = 2    
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="both", which="both", labelsize=14)
    ax.set_xlim([dates[0] - datetime.timedelta(days=delta + 5),
                 dates[-1] +  datetime.timedelta(days=delta)])    
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel(f"{metric} (\%)", rotation=90)
    ax.set_xlabel("$T_2$")

    x0 = -0.075
    ax.text(0.2, x0, "2020", transform=ax.transAxes, ha="center", va="center")
    ax.text(0.8, x0, "2021", transform=ax.transAxes, ha="center", va="center")
    
    ax.grid("on", which="both")
    ax.legend(bbox_to_anchor=(0.525, 0.625), fontsize=15)
    ax.grid("on", which="both", axis="both")
    ax.set_title(f"Error of {model_wrapper.prefix} on {model_wrapper.dataset_name}")
    
    
def print_DM_cross(dms, ms, versions):
    dfs = {}
    vkeys = [k for k in versions.keys()]
    for (k, country) in enumerate(ms.keys()):
        df = pandas.DataFrame()
        enriched_ind = np.where(ms[country][:, 0] == vkeys[0])[0]
        multi_ind = np.where(ms[country][:, 0] == vkeys[1])[0]

        df["Model"] = [k.string() for k in ms[country][enriched_ind, 1]]
        df[f"Pvalue({versions[vkeys[0]]}, {versions[vkeys[1]]})"] = np.round(dms[k, enriched_ind, multi_ind], decimals=4)
        df[f"Pvalue({versions[vkeys[1]]}, {versions[vkeys[0]]})"] = np.round(dms[k, multi_ind, enriched_ind], decimals=4)
        dfs[country] = df
    return dfs
        
def plot_DM_test_cross(dms, ms, version):
    countries = [k for k in ms.keys()]
    fig = plt.figure(figsize=(19.2, 10.8))
    gs1 = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)
    ax1 = fig.add_subplot(gs1[:, 0])
    ax2 = fig.add_subplot(gs1[:, 1], sharex=ax1)
    ax3 = fig.add_subplot(gs1[:, 2], sharex=ax1)

    axes = np.array([ax1, ax2, ax3])
    for i, country in enumerate(countries):
        data = dms[i, :, :]
        columns = [k.string() for k in ms[country][:, 1]]
        nmw = len(columns)
        ax = axes[i]
        data = np.where(np.isnan(data), 1, data)
        im = ax.imshow(data, cmap=cmap(), vmin=0, vmax=0.1)

        labels_fontsize = 14
        cols_to_display = [c.split(" ")[-1] for c in columns]
        ax.set_xticks(range(len(cols_to_display)))
        ax.set_xticklabels([])
        for x, col in enumerate(cols_to_display):
            ax.text(x + 0.25, nmw - 0.5, col, rotation=45,
                    fontsize=labels_fontsize, ha="right", va="top")
        
        ax.set_yticks(range(len(cols_to_display)))
        ax.set_yticklabels(cols_to_display, fontsize=labels_fontsize)
        ax.plot(range(len(cols_to_display)), range(len(cols_to_display)), 'wx')
        ax.set_title(country)

        i_masks = (2, 4, 6)
        x0 = -0.23
        for j in range(nmw):                       
            if j in i_masks:
                yaxes = 1 - (j / nmw)            
                ax.annotate("",
                            xy=[x0, yaxes],
                            xycoords="axes fraction",
                            xytext = [0, yaxes],                         
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                xaxes = 0.01 + j / nmw
                xratio = 1.25
                y0 = -0.15
                ax.annotate("",
                            xy=[xaxes, 0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                ax.annotate("",
                            xy=[xaxes - xratio / nmw, y0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0-0.05],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
        yindices = (3, 7)
        labels = ("SVR", "SVR")
        for yind, label in zip(yindices, labels):
            ax.text(-2.5, yind, label, ha="center", size=labels_fontsize,
                    rotation=90)

        xindices = (1.5, 5.5)
        labels = ("SVR", "SVR")
        for xind, label in zip(xindices, labels):
            pad = 1.5
            ax.text(xind, nmw + pad, label, ha="center", size=labels_fontsize)

        xindices = (0.5, 4.5)
        labels = version.values()
        pad = 2.25
        for xind, label in zip(xindices, labels):
            ax.text(xind, nmw + pad, label, ha="center", size=labels_fontsize,
                    c="r")

        yindices = (2.5, 6.5)
        labels = version.values()
        pad = 3.25
        for yind, label in zip(yindices, labels):
            ax.text(0 - pad, yind, label, ha="center", size=labels_fontsize,
                    c="r", rotation=90)            
            
        # Separate versions
        xsep = 0.5
        ysep = 0.5
        ax.annotate("",
                    xy=[-0.45, ysep],
                    xycoords="axes fraction",
                    xytext = [1, ysep],                         
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})
        ax.annotate("",
                    xy=[xsep, 0],
                    xycoords="axes fraction",
                    xytext = [xsep, 1],                         
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})
        xratio = 1.2
        ax.annotate("",
                    xy=[xsep, 0],
                    xycoords="axes fraction",
                    xytext = [xsep - xratio / nmw, y0],
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})
        ax.annotate("",
                    xy=[xsep - xratio / nmw, y0],
                    xycoords="axes fraction",
                    xytext = [xsep - xratio / nmw, y0-0.2],
                    textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                "linewidth" : 3, "color" : "r"})        
            
        ax.tick_params(axis="y", pad=-0.1)
        ax.set_xticklabels([])
        ax.set_xticks([c + 0.5 for c in range(len(columns))], minor=True)
        ax.set_yticks([c + 0.45 for c in range(len(columns))], minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid("on", which="minor", linestyle="--", c="grey")

    
    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05,
                        pad=0.21)
    cbar.ax.set_xlabel("Pvalue of the DM test")
        
    #plt.show()
    
    
def plot_DM_test(dms, model_wrappers, dataset, version):
    countries = [k for k in model_wrappers[version][dataset].keys()]
    columns, ind_lago = format_cols(
        model_wrappers[version][dataset][countries[0]])
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

    nmw = 12
    for i, country in enumerate(countries):
        data = dms[version][dataset][i, :, :]        
        ax = axes[i]
        data = np.where(np.isnan(data), 1, data)
        im = ax.imshow(data, cmap=cmap_2(), vmin=0, vmax=0.051)

        labels_fontsize = 14
        cols_to_display = [c.split(" ")[-1] for c in columns]
        ax.set_xticks(range(len(cols_to_display)))
        for x, col in enumerate(cols_to_display):
            ax.text(x + 0.25, nmw + 0.5, col, rotation=45,
                    fontsize=labels_fontsize, ha="right", va="top")
        
        ax.set_yticks(range(len(cols_to_display)))
        ax.set_yticklabels(cols_to_display, fontsize=labels_fontsize)
        ax.plot(range(len(cols_to_display)), range(len(cols_to_display)), 'wx')
        ax.set_title(country)

        i_masks = (3, 5, 9)
        x0 = -0.23
        for j in range(nmw):                       
            if j in i_masks:
                yaxes = 1 - (0.92*j / nmw)            
                ax.annotate("",
                            xy=[x0, yaxes],
                            xycoords="axes fraction",
                            xytext = [0, yaxes],                         
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                xaxes = 0.01 + 0.92*j / nmw
                xratio = 1.75
                y0 = -0.15
                ax.annotate("",
                            xy=[xaxes, 0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})
                ax.annotate("",
                            xy=[xaxes - xratio / nmw, y0],
                            xycoords="axes fraction",
                            xytext = [xaxes - xratio / nmw, y0-0.05],
                            textcoords="axes fraction",
                            arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                        "linewidth" : 1, "color" : "k"})

                
        yindices = (4.0, 7.1, 11.4)
        labels = ("SVR", "LEAR", "DNN")
        for yind, label in zip(yindices, labels):
            ax.text(-3, yind, label, ha="center", size=labels_fontsize,
                    rotation=90)

        xindices = (1.8, 4.8, 8.5)
        labels = ("SVR", "LEAR", "DNN")
        for xind, label in zip(xindices, labels):
            pad = 2.9
            ax.text(xind, nmw + pad, label, ha="center", size=labels_fontsize)
            
        ax.tick_params(axis="y", pad=-0.1)

        ax.set_xticklabels([])
        ax.set_xticks([c + 0.5 for c in range(len(columns))], minor=True)
        ax.set_yticks([c + 0.45 for c in range(len(columns))], minor=True)
        ax.tick_params(which="minor", length=0)
        ax.grid("on", which="minor", linestyle="--", c="grey")
            
    if len(countries) != 3:
        cbar = plt.colorbar(im, ax=axes[-1], extend="max")
        cbar.ax.set_ylabel("Pvalue of the DM test")
        cbar.ax.tick_params(labelsize=labels_fontsize)
    else:
        cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05)
        cbar.ax.set_xlabel("Pvalue of the DM test")
    fig.suptitle(f"DM test PVALUES for the recalibrated forecasts of the Test set")
    
    dfdata = {"Country" : [],
              "Model 1" : [],
              "Model 2" : [],
              "P Value 1" : [],
              "P Value 2" : []}
    for i, country in enumerate(countries):
        data = np.round(dms[version][dataset][i, :, :], decimals=3)
        print(data)
        model_comb = [k for k in itertools.combinations(
            range(data.shape[0]), 2)]
        m1s = [k[0] for k in model_comb]
        m2s = [k[1] for k in model_comb]
        mstr = [m.string() for m in  model_wrappers[version][dataset][country]]
        for (m1, m2) in zip(m1s, m2s):
            dfdata["Country"].append(country)
            dfdata["Model 1"].append(mstr[m1])
            dfdata["Model 2"].append(mstr[m2])
            dfdata["P Value 1"].append(data[m1, m2])
            dfdata["P Value 2"].append(data[m2, m1])
    df = pandas.DataFrame(dfdata)    
    return df

def plot_DM_test_BL(dms, model_wrappers, datasets, s="test_recalibrated"):
    country = [k for k in model_wrappers[datasets[0]][s].keys()][0]
    columns = [m.string() for m in model_wrappers[datasets[0]][s][country]]
    factor = 1
    fig, axes = plt.subplots(1, 3, figsize=(factor * 19.2, factor * 10.8))

    nmw = 5
    for i, dataset in enumerate(datasets):
        data = dms[dataset][s][0, :, :]        
        ax = axes[i]
        data = np.where(np.isnan(data), 1, data)
        im = ax.imshow(data, cmap=cmap_2(), vmin=0, vmax=0.051)

        labels_fontsize = 14
        cols_to_display = [c.split(" ")[-1] for c in columns]
        ax.set_xticks(range(len(cols_to_display)))
        for x, col in enumerate(cols_to_display):
            ax.text(x + 0.25, nmw + 0.5, col, rotation=45,
                    fontsize=labels_fontsize, ha="right", va="top")
        
        ax.set_yticks(range(len(cols_to_display)))
        ax.set_yticklabels(cols_to_display, fontsize=labels_fontsize)
        ax.plot(range(len(cols_to_display)), range(len(cols_to_display)), 'wx')
        ax.set_title(dataset)

    cbar = plt.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05)
    cbar.ax.set_xlabel("Pvalue of the DM test")
    fig.suptitle(f"DM test PVALUES for the recalibrated forecasts of the Test set")
    
    dfdata = {"Dataset" : [],
              "Model 1" : [],
              "Model 2" : [],
              "P Value 1" : [],
              "P Value 2" : []}
    for i, dataset in enumerate(datasets):
        data = np.round(dms[dataset][s][0, :, :], decimals=3)
        print(data)
        model_comb = [k for k in itertools.combinations(
            range(data.shape[0]), 2)]
        m1s = [k[0] for k in model_comb]
        m2s = [k[1] for k in model_comb]
        mstr = [m.string() for m in  model_wrappers[dataset][s][country]]
        for (m1, m2) in zip(m1s, m2s):
            dfdata["Dataset"].append(dataset)
            dfdata["Model 1"].append(mstr[m1])
            dfdata["Model 2"].append(mstr[m2])
            dfdata["P Value 1"].append(data[m1, m2])
            dfdata["P Value 2"].append(data[m2, m1])
    df = pandas.DataFrame(dfdata)    
    return df

def compute_pvalues_cross(countries, dataset, predictions, model_wrappers,
                          real_prices, versions):
    versions = versions.keys()
    models = np.concatenate([[(version, mw) for mw in model_wrappers[version][dataset][countries[0]] if ((type(mw) is not LAGOWrapper) and (type(mw) is not CNNWrapper))] for version in versions])
    n = len(models)  
    dms = np.zeros((len(countries), n, n))
    ms = {}
    for (k, country) in enumerate(countries):
        models = np.concatenate([[(version, mw) for mw in model_wrappers[version][dataset][country] if ((type(mw) is not LAGOWrapper) and (type(mw) is not CNNWrapper))] for version in versions])
        ms[country] = models
        n = len(models)    
        for (i, (v1, m1)) in enumerate(models):
            for (j, (v2, m2)) in enumerate(models):
                if i == j:
                    dms[k, i, j] = np.nan
                else:
                    p1 = predictions[v1][dataset][country][m1]
                    p2 = predictions[v2][dataset][country][m2]
                    if p1 is None or p2 is None:
                        dms[k, i, j] = np.nan
                    else:
                        dms[k, i, j] = DM(
                            p_real=real_prices[dataset][country],
                            p_pred_1=p1, p_pred_2=p2,
                            norm="mae", version=version)
    return dms, ms

def compute_pvalues(predictions, model_wrappers, real_prices,
                    version="multivariate"):
    datasets = [k for k in predictions.keys()] 
    dms = {}
    for dataset in datasets:
        sets = [k for k in predictions[dataset].keys()]         
        dms[dataset] = {}
        for s in sets:
            countries = [k for k in predictions[dataset][s].keys()]
            if len(countries) > 0:
                nmw = len(model_wrappers[dataset][s][countries[0]])
                dms[dataset][s] = np.zeros((len(countries), nmw, nmw))
                for (k, country) in enumerate(countries):
                    for (i, m1) in enumerate(model_wrappers[dataset][s][country]):
                        for (j, m2) in enumerate(model_wrappers[dataset][s][country]):
                            if i == j: dms[dataset][s][k, i, j] = np.nan
                            else:
                                p1 = predictions[dataset][s][country][m1]
                                p2 = predictions[dataset][s][country][m2]
                                preal = real_prices[s][country]

                                if len(p1.shape) == 1:
                                    preal = preal                                   
                                if p1 is None or p2 is None:
                                    dms[dataset][s][k, i, j] = np.nan
                                else:
                                    dms[dataset][s][k, i, j] = DM(
                                        p_real=preal, p_pred_1=p1, p_pred_2=p2,
                                        norm="mae", version=version)
    return dms

def plot_summary(res_1, res_2, res_3):
    final = res_1.join(res_2, how = "inner").join(res_3, how="inner")
    indices = final.index
    countries = np.unique(np.array([c[0] for c in indices]))
    metrics = np.unique(np.array([c[1] for c in indices]))
    
    ind_lago = np.array([("DNN " in i) or ("LEAR" in i) for i in final.columns])
    final = sort_columns(final, ind_lago)

    plot_matrix(
        final, data_info="Scaled metrics",
        title_info=f"All 3 versions",
        colormap=cmap_scaled_values(), scaler_class=MinAbsScaler)

    return final

def plot_summary_multi(res_1, res_2):
    final = res_1.join(res_2, how = "inner")
    indices = final.index
    countries = np.unique(np.array([c[0] for c in indices]))
    metrics = np.unique(np.array([c[1] for c in indices]))
    
    ind_lago = np.array([("DNN " in i) or ("LEAR" in i) for i in final.columns])
    final = sort_columns(final, ind_lago)

    plot_matrix(
        final, data_info="Scaled metrics",
        title_info=f"Models and their multi country counterpart",
        colormap=cmap_scaled_values(), scaler_class=MinAbsScaler)

    return final

def sort_columns(res, ind_lago):
    columns = res.columns
    indices = res.index
    
    columns_lago = copy.deepcopy(columns[ind_lago])
    ind_columns = np.argsort(columns[np.logical_not(ind_lago)])
    ind_columns = np.concatenate(
        (ind_columns,
         np.array([i for i in range(
             len(ind_columns), len(ind_columns) + len(columns_lago))], int)))
    sorted_columns = np.concatenate(
        (columns[np.logical_not(ind_lago)], columns_lago))[ind_columns]
        
    res = pandas.DataFrame(res.values, columns=columns, index=indices)
    res = res.loc[:, sorted_columns]
    return res

def plot_matrix(data, title_info="", data_info="", colormap=cmap_scaled_values(),
                scaler_class=MinAbsScaler, vmin=None, vmax=None):
    data = copy.deepcopy(data)
    
    # Normalize line by line
    if scaler_class is not None:
        for i in range(data.shape[0]):
            scaler = scaler_class()
            values = data.iloc[i].values.reshape(-1, 1)
            worse = values.max()
            values = np.where(values < 0, worse, values)
            data.iloc[i] = scaler.fit_transform(values).reshape(-1)
    plt.figure(figsize=(19.2, 10.8))
    dt = plt.imshow(data, cmap=colormap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(dt, extend="both")
    #cbar.ax.tick_params(direction="out", length=10, width=3, pad=8)
    #cbar.ax.set_ylabel(data_info, ha="center", labelpad=-128)
    cbar.ax.tick_params(direction="in", length=28, width=3, pad=-75)
    cbar.ax.set_ylabel(data_info, ha="center", labelpad=0)
    
    
    sorted_columns = data.columns
    plt.xticks(range(len(sorted_columns)), sorted_columns, rotation=45, ha="right", va="top")
    plt.tick_params(axis="x", pad=-5)
    
    index = data.index
    plt.yticks(range(len(index)), index)
    
    plt.title(f"{title_info}", y=1.025)
    # plt.show()

def format_cols(mws):
    versioning = lambda x: "" if x.dataset_name.split("_")[0][-1] == "F" else x.dataset_name.split("_")[0][-1]
    multi = lambda x: "" if x.dataset_name.split("_")[1] != "FRDEBE" else " MULTI"
    
    columns = np.array([mw.string() + versioning(mw) + multi(mw) for mw in mws])
    ind_lago = np.array([type(mw) is LAGOWrapper for mw in mws])

    columns = [c.replace("_", " ") for c in columns]
    return columns, ind_lago

def plot_diff(res_1, res_2, title_info):
    indices = res_1.join(res_2, how="inner").index
    countries = np.unique(np.array([c[0] for c in indices]))
    metrics = np.unique(np.array([c[1] for c in indices]))

    res_1 = res_1.loc[indices]
    res_2 = res_2.loc[indices]

    columns = []
    columns_2 = []
    for k1 in res_1.keys():
        for k2 in res_2.keys():
            if (k1 in k2) or (k2 in k1):
                columns.append(k1)
                columns_2.append(k2)

    res_1 = res_1[columns].values
    res_2 = res_2[columns_2].values
    
    diff  = pandas.DataFrame(100 * ((res_1 - res_2) / res_1), columns=columns,
                             index=indices)
    vabs = max(abs(np.quantile(diff, 0.05)), abs(np.quantile(diff, 0.95)))
    vabs = 15
    
    if "MULTI" in columns[0]: columns = [c.split("MULTI")[0] for c in columns]
    if "2" in columns[0]: columns = [c.split("2")[0] for c in columns]
    columns = [c.split(" ")[-1] for c in columns]

    diff.columns = columns
    diff.index = [c[-1] for c in diff.index]
    
    x, y = (1, 1)
    #plt.text(x + 2.9, y + 3, "Models 2 better than  Models 1", rotation=90)
    #plt.text(x + 2.9, y + 9, "Models 1 better than  Models 2", rotation=90)    
    plot_matrix(
        diff,
        data_info="$100\\frac{" + title_info[0][1:-1] + "-" + title_info[1][1:-1] + "}{" + title_info[0][1:-1] + "}\%$",        
        title_info=f"Comparing {title_info[0]} and {title_info[1]}",
        colormap=cmap_diff_values_2(), scaler_class=None, vmin=-vabs, vmax=vabs)

    nmw = len(columns)    
    plt.xticks(range(len(columns)), [])
    for x, col in enumerate(columns):
        plt.text(x + 0.25, len(metrics) * len(countries) - 0.55, col, rotation=45, ha="right", va="top")  
    
    y0 = len(columns) * len(countries) - 1
    x0 = -0.42
    for i, country in enumerate(countries):        
        plt.text(5.1 * x0, y0 + 0.5 - ((i + 1)  * len(metrics)  + len(metrics) / 2), country)
        if i != len(countries) - 1:
            yaxes = (i + 1) / len(countries)
            plt.annotate("",
                         xy=[x0, yaxes],
                         xycoords="axes fraction",
                         xytext = [1, yaxes],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 5, "color" : "k"})

    i_masks = (1, )
    y0 = -0.1
    for i in range(nmw):        
        xaxes = 0.01 + (i + 1) / nmw
        if i in i_masks:
            ratio = 0.95
            plt.annotate("",
                         xy=[xaxes, 0],
                         xycoords="axes fraction",
                         xytext = [xaxes - ratio / nmw, y0],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})
            plt.annotate("",
                         xy=[xaxes - ratio / nmw, y0],
                         xycoords="axes fraction",
                         xytext = [xaxes - ratio / nmw, y0-0.1],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})             
    xindices = (1.95, )
    labels = ("SVR", )
    for xind, label in zip(xindices, labels):
        plt.text(xind, len(countries) * len(metrics) + 0.8, label, ha="center")
    return diff

def plot_scaled_metrics(results, model_wrappers, dataset, version, metrics):
    countries = [k for k in model_wrappers[version][dataset].keys()]
    nmw = len(model_wrappers[version][dataset][countries[0]])
    data = np.zeros((len(countries) * len(metrics), nmw))
    if len(countries) == 3: epf = False
    else: epf = True

    for i in range(len(countries)):
        data[i * len(metrics):(i + 1) * len(metrics), :] = results[version][dataset][i, :, :].transpose()

    columns, ind_lago = format_cols(model_wrappers[version][dataset][countries[0]])
    indices = np.concatenate([np.array([m.__name__ for m in metrics]) for c in countries])
    
    #res = sort_columns(pandas.DataFrame(data, columns=columns, index=indices), ind_lago)
    res = pandas.DataFrame(data, columns=columns, index=indices)
    col_temp = copy.deepcopy(res.columns)
    if "MULTI" in res.columns[0]: res.columns = [c.split("MULTI")[0] for c in res.columns]    
    if "2" in res.columns[0]: res.columns = [c.split("2")[0] for c in res.columns]
    res.columns = [c.split(" ")[-1] for c in res.columns]
    
    plot_matrix(res, colormap=cmap_scaled_values(), scaler_class=MinAbsScaler,
                title_info="Scaled metrics on the\nRecalibrated Forecasts",
                data_info="Fraction of the best metric", vmin=1, vmax=1.25)
    
    cols_to_display = res.columns
    res.columns = copy.deepcopy(col_temp)
    
    res.index = [(c, m.__name__) for (c, m) in itertools.product(countries, metrics)]
    plt.xticks(range(len(cols_to_display)), [])
    pad = -0.45
    for x, col in enumerate(cols_to_display):
        plt.text(x + 0.35, len(metrics) * len(countries) + pad, col, rotation=52, ha="right", va="top")    

    y0 = len(metrics) * len(countries) - 1
    x0 = -0.5
    if epf:
        pad_text = 7
        pad_line = -0.25
    else:
        pad_text = 4
        pad_line = -0.4        
    for i, country in enumerate(countries):        
        plt.text(pad_text * x0, i * len(metrics) + len(metrics) / 2, country)
        if i != len(countries) - 1:
            yaxes = (i + 1) / len(countries)
            plt.annotate("",
                         xy=[pad_line, yaxes],
                         xycoords="axes fraction",
                         xytext = [1, yaxes],                         
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 5, "color" : "k"})

    if epf: i_masks = (2, 4, 8)
    else: i_masks = (2, )
    y0 = -0.1
    if epf: fraction = 1.1
    else: fraction = 0.7
    for i in range(nmw):        
        xaxes = 0.01 + (i + 1) / nmw
        if i in i_masks:
            plt.annotate("",
                         xy=[xaxes, 0],
                         xycoords="axes fraction",
                         xytext = [xaxes - fraction / nmw, y0],
                         textcoords="axes fraction",
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})
            plt.annotate("",
                         xy=[xaxes - fraction / nmw, y0],
                         xycoords="axes fraction",
                         xytext = [xaxes - fraction / nmw, y0-0.1],   
                         textcoords="axes fraction",                          
                         arrowprops={"arrowstyle": "-", "linestyle" : "--",
                                     "linewidth" : 3, "color" : "k"})
            
            
    if epf:
        pad = 1.5
        xindices = (2.25, 5.25, 10.5)
        labels = ("SVR", "LEAR", "DNN")
    else:
        pad = 0.7
        xindices = (1.95, )
        labels = ("SVR", )        
    for xind, label in zip(xindices, labels):
        plt.text(xind, len(countries) * len(metrics) + pad, label, ha="center")
    
    return res    

def compute_metrics(predictions, model_wrappers, metrics, real_prices, naive_forecasts):
    results = {}
    datasets = [k for k in predictions.keys()]
    for dataset in datasets:
        results[dataset] = {}
        sets = [k for k in predictions[dataset].keys()]
        for s in sets:
            countries = [k for k in predictions[dataset][s].keys()]
            if len(countries) > 0:
                nmw = len(model_wrappers[dataset][s][countries[0]])
                results[dataset][s] = -np.ones((len(countries), nmw, len(metrics)))
                for (i, country) in enumerate(countries):
                    y_true = real_prices[s][country]
            
                    for (j, model_wrapper) in enumerate(
                            model_wrappers[dataset][s][country]):
                        y_pred = predictions[dataset][s][country][model_wrapper]
                        if y_pred is not None:
                            for (k, metric) in enumerate(metrics):
                                if metric == rmae:
                                    value = metric(y_true, y_pred,
                                                   naive_forecasts[s][country])
                                else: value = metric(y_true, y_pred)                
                                results[dataset][s][i, j, k] = value
    
    return results

def load_forecasts(sets, countries, models, datasets, lago_params=None):
    # Load all predictions
    predictions = {}
    model_wrappers = {}
    for dataset in datasets:
        predictions[dataset] = {}
        model_wrappers[dataset] = {}        
        for s in sets:            
            predictions[dataset][s] = {}
            model_wrappers[dataset][s] = {}
            if dataset != "3":
                for country in countries:
                    predictions[dataset][s][country] = {}   
                    model_wrappers[dataset][s][country] = []              
                    for model in models:
                        if model != LAGOWrapper:
                            name = model.string(model) + "_TSCHORA"
                            model_wrappers_temp = [model(
                                name, f"EPF{dataset}_{country}", "")]
                        else:
                            model_wrappers_temp = []
                            for key in lago_params.keys():
                                for value in lago_params[key]:
                                    name = f"{key}_{str(value)}"
                                    model_wrappers_temp.append(model(
                                        name, f"EPF{dataset}_{country}", ""))
                        
                        for model_wrapper in model_wrappers_temp:
                            try:
                                all_prevs = load_prevs_mw(model_wrapper, s).values
                            except: all_prevs = None
                            if all_prevs is not None:
                                predictions[dataset][s][country][model_wrapper] = all_prevs
                                model_wrappers[dataset][s][country].append(
                                    model_wrapper)
                                
                    if predictions[dataset][s][country] == {}:
                        del predictions[dataset][s][country]
                        del model_wrappers[dataset][s][country]

            if dataset == "3":
                for (i, cc) in enumerate(("FR", "DE", "BE")):
                    predictions[dataset][s][cc] = {}
                    model_wrappers[dataset][s][cc] = []
                    for model in models:
                        name = model.string(model) + "_TSCHORA"
                        model_wrapper = model(name, "EPF2_FRDEBE", "")
                        try:
                            all_prevs = load_prevs_mw(model_wrapper, dataset).values
                        except Exception as e:
                            #print(e)
                            all_prevs = None

                        if all_prevs is not None:
                            predictions[dataset][s][cc][model_wrapper] = {}
                            ypred = all_prevs[:, 24*i:24*(i+1)]
                            predictions[dataset][s][cc][model_wrapper] = ypred
                            model_wrappers[dataset][s][cc].append(model_wrapper)

                    if predictions[dataset][s][cc] == {}:
                        del predictions[dataset][s][cc]
                        del model_wrappers[dataset][s][cc]

    return predictions, model_wrappers

def load_real_prices(countries, model_wrappers, dataset="", hourly=True):    
    real_prices = {"train" : {}, "test" : {}, "validation": {},
                   "test_recalibrated" : {}}
    naive_forecasts = {"train" : {}, "test" : {}, "validation": {},
                       "test_recalibrated" : {}}
    for country in countries:
        if country == "FRDEBE": pass
        else:
            if dataset in ("", "2", "3"): nval = 362
            else: nval = 365       
            naive_wrapper = Naive("NAIVE", f"{dataset}{country}",
                                  country, hourly=hourly,
                                  spliter=MySplitter(nval, shuffle=False))
            if hourly:
                f = lambda x: x
            else:
                f = lambda x: x.mean(axis=1).reshape(-1, 1)
                        
            # Need to extract the dates of the sets
            k = [k for k in model_wrappers.keys()][0]
            s = [k for k in model_wrappers[k]][0]
            mw = model_wrappers[k][s][country][0]
            train_dates = pandas.read_csv(
                mw.train_dataset_path()).period_start_date.values
            test_dates = pandas.read_csv(
                mw.test_dataset_path()).period_start_date.values

            naive_train_dates = pandas.read_csv(
                naive_wrapper.train_dataset_path()).period_start_date.values
            naive_test_dates = pandas.read_csv(
                naive_wrapper.test_dataset_path()).period_start_date.values

            # Compute the indices of the dates to select in the NAIVE sets
            train_indices = [d in train_dates for d in naive_train_dates]
            test_indices = [d in test_dates for d in naive_test_dates]
            
            X, y = naive_wrapper.load_train_dataset()
            Xt, yt = naive_wrapper.load_test_dataset()            

            # Filter the dates not present in the operational dataset
            X = X[train_indices, :]
            y = y[train_indices, :]

            Xt = Xt[test_indices, :]
            yt = yt[test_indices, :]
            
            # Need to re-split for taking the validation prices
            ((Xtr, ytr), (Xv, yv)) = naive_wrapper.spliter(X, y)
            pandas.DataFrame(yv).to_csv(country + ".csv")
            real_prices["validation"][country] = f(yv)
            real_prices["train"][country] = f(ytr)
            real_prices["test"][country] = f(yt)
            real_prices["test_recalibrated"][country] = f(yt)
            
            # Also computes the naive forecasts    
            naive_forecasts["validation"][country] = naive_wrapper.predict(None, Xv)
            naive_forecasts["train"][country] = naive_wrapper.predict(None, Xtr)
            naive_forecasts["test"][country] = naive_wrapper.predict(None, Xt)
            naive_forecasts["test_recalibrated"][country] = naive_wrapper.predict(
                None, Xt)

    return real_prices, naive_forecasts

def graph_metrics(models):
    model_wrappers = []
    Maes = []
    Smapes = []
    Accs = []
    for model in models:        
        model_wrapper = model(f"{model.string(model)}_TSCHORA", "AllEuropeGNN",
                              countries_to_predict="all",
                              known_countries=["CH", "GB"],
                              framework="GraphLevel",
                              spliter=MySplitter(365, shuffle=False))
        model_wrappers.append(model_wrapper)
        X, y = model_wrapper.load_train_dataset()
        maes, smapes, accs = model_wrapper.compute_metrics(seg=10)
        Maes.append(maes)
        Smapes.append(smapes)
        Accs.append(accs)
    return Maes, Smapes, Accs

def df_to_latex(df, index=True, index_col="", hlines=True, roundings=[],
                highlight=[]):    
    headers = [index_col] + list(df.columns)
    nc = len(headers)
    col_params = "|"
    tab_cols = ""
    for header in headers:
        tab_cols += "\\textbf{" + header + "} & "
        col_params += "c|"
    tab_cols = tab_cols[:-2]

    if highlight == []:
        highlight = ["" for header in headers]

    if roundings == []:
        roundings = [2 for header in headers]
    try:
        roundings[0]
    except:
        roundings = [roundings for header in headers]

    s = ""
    # DEFINE TABULAR 
    s += """
    \\begin{table}[htb]
    \\begin{center}
          \scalebox{1}{
            \\begin{tabular}{"""
    s += col_params
    s += """}
              \hline
    """
    s += tab_cols + "\\\\"
    s += """
              \hline
    """
    if hlines:
        s += """
                  \hline
        """

    rows = list(df.index)
    nr = len(rows)
    for row in rows:
        # Add the index in first column
        i_row = str(row)        
        if index:
            i_row = "\\textbf{" + i_row + "}"
        s += i_row
        values = df.loc[row].values
        for v, highlight_, col, rounding in zip(
                values, highlight, df.columns, roundings):
            str_v = str(round(v, ndigits=rounding))
            if str_v == "nan": str_v = " - "
            str_v_highlighted = "{\\bf "+ str_v + "}"
            if ((highlight_ == "high") and (v == max(df.loc[:, col]))) or ((highlight_ == "low") and (v == min(df.loc[:, col]))):
                str_v = str_v_highlighted
            s += " & " + str_v
        s += """\\\\
        """
        if hlines:
            s += """\hline
            """

    if not hlines: s+= """\hline
    """
    s +=  """ \end{tabular}  
    }
    \end{center}
    \caption{}
    \label{}
    \end{table}"""
    return s
    


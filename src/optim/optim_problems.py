import numpy as np, pandas, time, os, datetime, itertools, copy
from scipy.optimize import linprog, least_squares

import matplotlib.pyplot as plt, matplotlib
import matplotlib.ticker as tck
import matplotlib.dates as mdates

from src.models.GNNWrapper import GNNWrapper
from src.analysis.metrics import CC, mae, smape, DM
from src.optim.flow_utils import *

####################### I/O functions
def load_data(dataset):
    model_wrapper = NodeGNNWrapper("FLOWS", "", known_countries=[])

    if dataset == "train":
        X, Y = model_wrapper.load_train_dataset()
    else:
        X, Y = model_wrapper.load_test_dataset()
        
    ptemp = model_wrapper.params()
    ptemp["n_epochs"] = 1
    regr = model_wrapper.make(ptemp)
    regr.fit(X, Y)
    model = regr.regressor_.steps[1][1]
    zones = model.country_list    
    dates, Xe, Xn = model.split_node_edge(X)
    datesh = [datetime.datetime(d.year, d.month, d.day, h)
              for d in dates for h in range(24)]
    
    edges, all_out, all_in = load_edges(model)
    network = (zones, edges, all_out, all_in)
    data = (dates, Xn, Xe)
    
    dims = problem_dims(Xn, zones, edges)
    return datesh, data, model, network, dims
    
def load_edges(model):
    all_edges = list(model.edges_columns_idx.keys())
    edges = np.array([e for e in all_edges if len(model.edges_columns_idx[e]) > 0])
    all_out = np.array([c.split("_")[0] for c in edges])
    all_in = np.array([c.split("_")[-1] for c in edges])
    return edges, all_out, all_in

def load_prices(data, network, model, dims):
    dates, Xn, Xe = data
    n, nz, T = dims
    zones, edges, all_out, all_in = network
    
    Pz = np.zeros((n, T))
    Pz_prime = np.zeros((n, T))    
    j = 0
    for i, c in enumerate(zones):
        conn = edges[np.where(all_out == c)[0]]
        prices_in = model.converted_node_indices[c][range(72, 96)]
        for con in conn:
            c_to = con.split("_")[1]
            prices_out = model.converted_node_indices[c_to][range(72, 96)]        
            Pz[:, j] = Xn[:, prices_in].reshape(-1)
            Pz_prime[:, j] = Xn[:, prices_out].reshape(-1)
            j += 1
    return Pz, Pz_prime

def load_fundamentals(data, network, model, dims):
    dates, Xn, Xe = data
    n, nz, T = dims
    zones, edges, all_out, all_in = network
    
    Cz = np.zeros((n, nz))
    Rz = np.zeros((n, nz))
    Gz = np.zeros((n, nz))    
    for i, c in enumerate(model.country_list):
        consumption = model.converted_node_indices[c][range(0, 24)]
        Cz[:, i] = Xn[:,consumption].reshape(-1)
        
        renewables_production = model.converted_node_indices[c][range(48, 72)]
        Rz[:, i] = Xn[:, renewables_production].reshape(-1)

        total_production = model.converted_node_indices[c][range(24, 48)]
        Gz[:, i] = Xn[:, total_production].reshape(-1)    

    return Cz, Rz, Gz

def load_maximal_generation_capacities(datesh, network):
    zones, edges, all_out, all_in = network    
    power_max = pandas.read_csv(os.path.join(
        os.environ["OPALE"], "data", "datasets", "maximal_generation_capacity.csv"))
    power_max.index = [datetime.datetime.strptime(
        d, "%Y-%m-%dT%H:00:00.0") for d in power_max.period_start_time]
    power_max = power_max.loc[datesh]
    power_max.drop(columns="period_start_time", inplace=True)
    power_max = power_max.loc[:, [c + "_PowerMax" for c in zones]]
    Vz = power_max.values
    return Vz

def load_network_constraints(data, network, model, dims):
    dates, Xn, Xe = data
    n, nz, T = dims
    zones, edges, all_out, all_in = network
    
    Az_zprime = np.zeros((n, T))
    j = 0
    for i, c in enumerate(zones):
        conn = edges[np.where(all_out == c)[0]]    
        for con in conn:
            edge_indices = np.array(model.converted_edge_indices[con])
            Az_zprime[:, j] = Xe[:, edge_indices].reshape(-1)
            j += 1
    return Az_zprime

def load(name):
    F = pandas.read_csv(
        os.path.join(os.environ["OPALE"],"data","datasets",f"{name}.csv"))
    F.period_start_date = np.array(
        [datetime.datetime.strptime(td, "%Y-%m-%d").date()
         for td in F.period_start_date])
    F.set_index("period_start_date", inplace=True)
    return F

def save_flows(Flin, Flsq, Fcmb, Funi, Flin_test, Flsq_test, Fcmb_test, Funi_test):
    trains = [Flin, Flsq, Fcmb, Funi]
    tests = [Flin_test, Flsq_test, Fcmb_test, Funi_test]
    names= ("Flin", "Flsq", "Fcmb", "Funi")
    
    for train, test, name in zip(trains, tests, names):
        res = pandas.concat([train, test])
        
        res.to_csv(
            os.path.join(os.environ["OPALE"], "data", "Optim", f"{name}.csv"))
        
##################### Problem formulations and resolution
def problem_dims(Xn, zones, edges):
    # Compute the number of connections
    T = len(edges)
    n = Xn.shape[0] * 24
    nz = len(zones)
    dims = (n, nz, T)
    return dims

####### Flin
def formulate_lin_problem(Pz, Pz_prime, Cz, Rz, Vz, Az_zprime, network, model,dims):
    n, nz, T = dims
    zones, edges, all_out, all_in = network
    
    # Weigth of the optiomization variables in the loss
    S = np.zeros((n, T+nz)) # The Generation does nothing on the cost
    S[:, :T] = -1

    # Constraints matrix on the optimization variables
    CD = np.zeros((nz, T)) # Flow matrix
    for i, c in enumerate(model.country_list):
        # Substract the outcomming flows to the energy balance        
        CD[i, np.where(all_out == c)[0]] = 1

        # ADD the incomming flows to the energy balance        
        CD[i, np.where(all_in == c)[0]] = -1    

    M = np.zeros((nz, T+nz))
    
    # Constraints on the flows    
    M[0:nz, 0:T] = copy.deepcopy(CD) 

    # Constraints on the generation : add it to the energy balance    
    M[0:nz, T:T+nz] = np.identity(nz) 

    # Fixed part of the constaints (independant of the optimization variables)
    b = Cz - Rz

    # Optimization variable bounds
    B = np.zeros((n, 2, T+nz))
    B[:, 1, 0:T] = Az_zprime
    B[:, 1, T:T+nz] = Vz
    return S, M, b, B

def compute_Flin(S, M, b, B, network, dims, datesh, data):
    n, nz, T = dims
    zones, edges, all_out, all_in = network
    dates, _, _ = data
    Flin = np.zeros((len(dates), 24 * T))

    # Solve the linear problem for all hours
    for h in range(n):
        if h % 500 == 0: print(f"Iteration {h}/{n}")
        results = linprog(S[h, :], A_eq=M, b_eq=b[h, :],bounds=B[h,:,:].transpose())
        if results.status == 2:
            print(f"FAILED for {h} :", results.message)
        else:
            current_hour = datesh[h].hour
            current_day = datesh[h].date()
            
            row_index = np.where(dates == current_day)[0][0]
            columns_indices = np.array([current_hour + 24 * i for i in range(T)])
            Flin[row_index, columns_indices] = copy.deepcopy(results.x[0:T])

    cols = [f"ATC_from_{con.split('_')[0]}_to_{con.split('_')[1]}_{h}"
            for con in edges for h in range(24)]
    Flin = pandas.DataFrame(Flin, index=dates, columns=cols)
    return Flin

###### Flsq
def formulate_lsq_problem(Pz, Pz_prime, Cz, Rz, Gz, Vz, Az_zprime,
                          network, model, dims):
    n, nz, T = dims
    zones, edges, all_out, all_in = network
    
    # Constraints matrix on the optimization variables
    CD = np.zeros((nz, T)) # Flow matrix
    for i, c in enumerate(model.country_list):
        # Add the outcomming flows to the energy balance        
        CD[i, np.where(all_out == c)[0]] = 1

        # Subtract the incomming flows to the energy balance        
        CD[i, np.where(all_in == c)[0]] = -1

    # Part of the energy balance independant of the optimization variables
    b = Cz - Rz - Gz

    # Optimization variable bounds
    B = np.zeros((n, 2, T))
    B[:, 1, 0:T] = Az_zprime + 0.01 # Ensure feasible solution space

    # Cost function to minimize
    def f(x, CD=None, b=None):    
        return np.dot(CD, x) + b

    x0 = np.zeros(T) # Starting point for the solution search
    return f, x0, CD, B, b

def compute_Flsq(f, x0, CD, B, b, network, dims, datesh, data):
    n, nz, T = dims
    zones, edges, all_out, all_in = network
    dates, _, _ = data    
    Flsq = np.zeros((len(dates), 24 * T))

    # Solve the least square problems for all hours
    for h in range(n):
        if h % 100 == 0: print(f"Iteration {h}/{n}")
        try:
            results = least_squares(f, x0, bounds=B[h], kwargs={"CD":CD, "b":b[h]})
        except:
            print(f"Failure for iteration {h}")
        else:        
            current_hour = datesh[h].hour
            current_day = datesh[h].date()
            
            row_index = np.where(dates == current_day)[0][0]
            columns_indices = np.array([current_hour + 24 * i for i in range(T)])
            Flsq[row_index, columns_indices] = copy.deepcopy(results.x)

            # Update starting point for optimal solution search
            if h == n-1:
                x0 = np.zeros(T)
            else:
                x0=copy.deepcopy(np.clip(results.x,a_min=B[h+1, 0],a_max=B[h+1, 1]))
            
    cols = [f"ATC_from_{con.split('_')[0]}_to_{con.split('_')[1]}_{h}"
            for con in edges for h in range(24)]
    Flsq = pandas.DataFrame(Flsq, index=dates, columns=cols)
    return Flsq

############ Fcmb and Funi
##### Utilities function
def compute_errors(F, Flin, Flsq, dates):
    dfs = {"Flin" : Flin, "Flsq" : Flsq, "F" : F}
    select_dates(dfs, dates)
    
    # Compute order, sort, reshape
    order, edges = compute_order(dfs["Flin"])
    sort_dfs(dfs, order)
    matrices = matrixize(dfs)

    metrics = [CC, mae, smape]    
    yhats = [matrices[k] for k in list(matrices.keys()) if k != "F"]
    ytrue = matrices["F"]
    res = np.zeros((len(metrics), len(yhats), ytrue.shape[1]))
    for i, m in enumerate(metrics):
        for j, yhat in enumerate(yhats):
            res[i, j] = m(ytrue, yhat, mean=False)

    res_mean = pandas.DataFrame(
        index = [k for k in matrices.keys() if k != "F"],
        columns = [m.__name__ for m in metrics],
        data = res.mean(axis=2).transpose())
    return res_mean

def compute_order(ATC, s="ATC_"):
    order = np.sort(np.array([c.split(s)[1] for c in ATC.columns]))
    o1 = []
    o2 = []
    for o in order:
        c1, c2 = (o.split("_")[1], o.split("_")[3])
        if ((c1, c2) not in o1) and ((c2, c1) not in o2) and ((c1, c2) not in o2) and ((c2, c1) not in o1):
            o1.append((c1, c2))
            o2.append((c2, c1))
    edges = np.array(o1 + o2)
    order = [ch(c1, c2, h) for c1, c2 in edges for h in range(24)]
    return order, edges

def sort_dfs(dfs, order):
    for df in dfs.keys():
        if len(dfs[df].columns[0].split("_")) > 5:
            dfs[df].columns = [c.split("_", 1)[1] for c in dfs[df].columns]
            dfs[df] = dfs[df].reindex(columns=order)
            
def matrixize(dfs):
    ret = {}
    for df in dfs.keys():
        x, y = dfs[df].values.shape
        res = np.zeros((int(x * 24), int(y / 24)))
        for c in range(int(y / 24)):
            res[:, c] = dfs[df].values[:, c*24:(c + 1)*24].reshape(-1)
        ret[df] = res
    return ret

def common_dates(dfs):
    k = list(dfs.keys())[0]
    dates = set(dfs[k].index)
    if len(dfs) > 1:
        for kprime in list(dfs.keys())[1:]:
            dates = dates.intersection(set(dfs[kprime].index))
    return np.sort(np.array(list(dates)))

def select_dates(dfs, dates):
    for k in dfs.keys():
        dfs[k] = dfs[k].loc[dates]

##### Fcmb
def compute_corrcoeffs(dfs):
    columns = [
        "conso_from", "conso_to", "conso_diff",
        "prod_from", "prod_to", "prod_diff",
        "prog_from", "prog_to", "prog_diff",        
        "rload_from", "rload_to", "rload_diff",
        "res_from", "res_to", "res_diff",        
        "discrete_rload", "hour", "weekday", "month"
    ]
    label = "error_diff"
    edges = list(dfs.keys())
    coeffs = np.zeros((len(columns), len(edges)))
    for (j, key) in enumerate(edges):
        df = dfs[key]
        for (i, col) in enumerate(columns):
            coeffs[i, j] = np.corrcoef([df.loc[:, col], df.loc[:, label]])[1, 0]

    return coeffs, columns

def modify_columns(columns):
    new_cols = []
    for col in columns:
        if col.split("_")[0] in ("rload", "conso", "prod", "res", "prog"):
            col_ = "discrete_" + col
        else:
            col_ = col
        new_cols += [col_]
    return np.array(new_cols)

def find_unique_rules(all_found):
    # Count occurences of each rule
    unique_rules = {}
    keys = list(all_found.keys())
    for k in keys:
        for found in all_found[k]:
            if found.index.name not in list(unique_rules.keys()):
                unique_rules[found.index.name] = found.index
            else:
                unique_rules[found.index.name] = np.unique(np.concatenate(
                    (unique_rules[found.index.name], found.index)))

    for key in list(unique_rules.keys()):
        unique_rules[key] = pandas.DataFrame(
            columns=["count", "sum"], index=unique_rules[key],
            data=np.zeros((len(unique_rules[key]), 2)))
        
    return unique_rules

def count_unique_rules(all_found, unique_rules):
    for key in list(all_found.keys()):
        for found in all_found[key]:
            unique_rules[found.index.name].loc[found.index, "count"] += 1
            unique_rules[found.index.name].loc[found.index, "sum"] += found.values.mean()   

def check_when_wp_better(df, column, columns, threshold=0.0, last=True):
    series = []
    for col in columns:
        series += [df.groupby(col).apply(lambda x: np.mean(x.loc[:, column]))]
    
    alls = df.groupby(
        ["is_weekend", "season", "peak_hour", "discrete_rload_diff"]).apply(
            lambda x: np.mean(x.loc[:, column]))

    if not last: series = series[:-1]
    res = []
    for serie in series:
        better = serie.loc[serie > threshold]
        if len(better.values) > 0:
            res += [better]

    return res

def country_dfs(info_dfs, edges):
    by_countries = pandas.DataFrame(
        columns = ["country_from", "country_to", "mean_diff"])
    for i, (c1, c2) in enumerate(edges):
        df = info_dfs[ch(c1, c2)]
        df_temp = pandas.DataFrame(
            index = [i],
            columns=["country_from", "country_to", "mean_diff"],
            data = np.array([c1, c2, df.error_diff.mean()],
                            dtype=object).reshape(1, 3))
        by_countries = pandas.concat((by_countries, df_temp))
    return by_countries

def create_rules(all_found, edges_wp_ls):
    # Summary of the rules
    rules = {}
    for found in list(all_found.keys()):
        rules[found] = [serie.index for serie in all_found[found]]

    # Allow to overwrite the entiere rules for this edge
    for c1, c2 in edges_wp_ls:
        rules[ch(c1, c2)] = [pandas.Index(
            range(24), dtype='int64', name='hour')]
    return rules

def discretize(res, columns, train_data=None):
    for col in columns:
        y = res.loc[:, col].values
        dest_col = f"discrete_{col}"
        if train_data is None:
            bins = [np.quantile(y, f) for f in [i/100 for i in range(0,100,1)]]
        else:
            train_serie = train_data.loc[:, col].values
            bins = [np.quantile(train_serie, f)
                    for f in [i/100 for i in range(0,100,1)]]
                        
        res.loc[:, dest_col] = np.digitize(y, bins=bins)

def get_edge_df(c1, c2, dfs, fundamentals, edges, train_data=None):
    conso = fundamentals["conso"]
    prod = fundamentals["renewable_prod"]
    prog = fundamentals["programmable_prod"]    
    
    columns_1 = [ch(c1, c2, h) for h in range(24)]
    columns_2 = [ch(c2, c1, h) for h in range(24)]

    conso_col_1 = [f"{c1}_consumption_{h}"for h in range(24)]
    prod_col_1 = [f"{c1}_renewables_production_{h}"for h in range(24)]
    prog_col_1 = [f"{c1}_production_{h}"for h in range(24)]    

    conso_col_2 = [f"{c2}_consumption_{h}"for h in range(24)]
    prod_col_2 = [f"{c2}_renewables_production_{h}"for h in range(24)]
    prog_col_2 = [f"{c2}_production_{h}"for h in range(24)]     
    
    conso_1 = conso.loc[:, conso_col_1].values.reshape(-1)
    conso_2 = conso.loc[:, conso_col_2].values.reshape(-1)

    prod_1 = prod.loc[:, prod_col_1].values.reshape(-1)
    prod_2 = prod.loc[:, prod_col_2].values.reshape(-1)

    prog_1 = prog.loc[:, prog_col_1].values.reshape(-1)
    prog_2 = prog.loc[:, prog_col_2].values.reshape(-1)    
    res = pandas.DataFrame()

    # Fundamentals
    res.loc[:, "conso_from"] = conso_1
    res.loc[:, "conso_to"] = conso_2
    res.loc[:, "conso_diff"] = conso_1 - conso_2
    res.loc[:, "prod_from"] = prod_1
    res.loc[:, "prod_to"] = prod_2
    res.loc[:, "prod_diff"] = prod_1 - prod_2    
    res.loc[:, "prog_from"] = prog_1
    res.loc[:, "prog_to"] = prog_2
    res.loc[:, "prog_diff"] = prog_1 - prog_2
    res.loc[:, "rload_from"] = res.conso_from - res.prod_from
    res.loc[:, "rload_to"] = res.conso_to - res.prod_to
    res.loc[:, "rload_diff"] = res.loc[:, "rload_from"] - res.loc[:, "rload_to"]
    
    # Date info
    res.loc[:, "period_start_date"] = [d for d in dfs["Flin"].index
                                       for h in range(24)]    
    res.loc[:, "hour"] = np.repeat(
        np.arange(24).reshape(-1, 1),
        len(dfs["Flin"].index), axis=1).reshape(-1, order="F")
    res.loc[:, "peak_hour"]= [((h > 8) and (h < 20)) for h in res.hour]    
    res.loc[:, "weekday"] = [d.weekday() for d in res.period_start_date]
    res.loc[:, "is_weekend"] = [d in (5, 6) for d in res.weekday]    
    res.loc[:, "month"] = [d.month for d in res.period_start_date]
    res.loc[:, "season"] = np.digitize([(m % 12) for m in res.month],
                                       bins=[0, 3, 6, 9])    

    # Errors
    res.loc[:, "F"] = dfs["F"].loc[:, columns_1].values.reshape(-1, 1)
    res.loc[:, "Flin"] = dfs["Flin"].loc[:, columns_1].values.reshape(-1, 1)
    res.loc[:, "Flsq"] = dfs["Flsq"].loc[:,columns_1].values.reshape(-1,1)
    res.loc[:, "error_wp"] = np.abs(res.F - res.Flin)
    res.loc[:, "error_ls"] = np.abs(res.F - res.Flsq)
    res.loc[:, "error_diff"] = res.error_ls - res.error_wp

    # Compute the residuals    
    indices_cfrom = edges[:, 0] == c1
    ls_from_cfrom = edges[indices_cfrom, 1]
    ls_cfrom = np.zeros(dfs['Flsq'].shape[0] * 24)
    for c2_ in ls_from_cfrom:
        cols_ls_from_cfrom=[ch(c1, c2_, h)  for h in range(24)]
        cols_ls_to_cfrom=[ch(c2_, c1, h) for h in range(24)]
        ls_cfrom += (dfs['Flsq'].loc[:, cols_ls_from_cfrom].values. reshape(-1, 1).ravel() - dfs['Flsq'].loc[:, cols_ls_to_cfrom].values.reshape(-1, 1).ravel())

    residual_from = res.rload_from - res.prog_from + ls_cfrom
    indices_cto = edges[:, 0] == c2
    ls_from_cto = edges[indices_cto, 1]
    ls_cto = np.zeros(dfs['Flsq'].shape[0] * 24)
    for c2_ in ls_from_cto:
        cols_ls_from_cto=[ch(c2, c2_, h)  for h in range(24)]
        cols_ls_to_cto=[ch(c2_, c2, h) for h in range(24)]
        ls_cto += (dfs['Flsq'].loc[:, cols_ls_from_cto].values.reshape(-1, 1).ravel() - dfs['Flsq'].loc[:, cols_ls_to_cto].values.reshape(-1, 1).ravel())

    residual_to = res.rload_to - res.prog_to + ls_cto

    res.loc[:, "res_from"] = residual_from
    res.loc[:, "res_to"] = residual_to
    res.loc[:, "res_diff"] = residual_from - residual_to
    
    # Discretize
    columns = ["conso_from", "conso_to", "conso_diff",
               "prod_from", "prod_to", "prod_diff",
               "prog_from", "prog_to", "prog_diff",               
               "rload_from", "rload_to", "rload_diff",
               "res_from", "res_to", "res_diff"]
    discretize(res, columns, train_data=train_data)
    y = res.loc[:, 'rload_diff'].values
    res.loc[:, 'discrete_rload'] = np.digitize(
        y, bins=[np.quantile(y, f) for f in [i/10 for i in range(0,10,2)]])
    return res

def compute_rules(F, Flin, Flsq, dates):
    # Load consumption forecast and intermittent production forecast
    fundamentals = {
        "conso" : load("consumption"),
        "renewable_prod" : load("renewable_generation"),
        "programmable_prod" : load("programmable_generation")
    }
    select_dates(fundamentals, dates)

    dfs = {"Flin" : Flin, "Flsq" : Flsq, "F" : F}
    select_dates(dfs, dates)
    order, edges = compute_order(dfs["Flin"])
    sort_dfs(dfs, order)

    # Compute the quantiles of the fundamentals
    info_dfs = {}
    for c1, c2 in edges:
        info_dfs[ch(c1, c2)] = get_edge_df(c1, c2, dfs, fundamentals, edges)

    # Compute the loss difference per quantile
    coeffs, columns = compute_corrcoeffs(info_dfs)    
    new_cols = modify_columns(columns)
    all_found = {}
    for c1, c2 in edges:
        df = info_dfs[ch(c1, c2)]
        res = check_when_wp_better(df,"error_diff",new_cols,threshold=25,last=False)
        if len(res) > 0:
            all_found[ch(c1, c2)] = res
        
    # Merge timestamps where Ldiff > 0
    unique_rules = find_unique_rules(all_found)
    count_unique_rules(all_found, unique_rules)

    # Compute Ldiff for an entiere zone
    by_countries = country_dfs(info_dfs, edges)
    links = by_countries.loc[by_countries.mean_diff > 25]
    edges_wp_ls = list(zip(links.country_from.values, links.country_to.values))

    # Format all (z, zprim, x, q) where Ldiff > 0
    rules = create_rules(all_found, edges_wp_ls)

    return info_dfs, rules

def apply_rules(dfs, info_dfs, rules):
    res = copy.deepcopy(dfs['Flsq'])
    upgraded_points = 0
    for edge in list(rules.keys()):
        df = info_dfs[edge]
        rules_edge = rules[edge]
        c1 = edge.split("_")[1]
        c2 = edge.split("_")[3]
        for rule in rules_edge:
            if rule.name in ("hour", "weekday", "month"):            
                if rule.name == "hour":
                    columns = [ch(c1, c2, h) for h in rule]
                    indices = res.index
                else:
                    columns = [ch(c1, c2, h) for h in range(24)]
                    if rule.name == "weekday":                
                        indices= np.array(
                            [d for d in res.index if d.weekday() in rule])
                    if rule.name == "month":
                        indices=np.array(
                            [d for d in res.index if d.month in rule])
                res.loc[indices, columns]=dfs['Flin'].loc[indices, columns]
                upgraded_points += len(columns) * len(indices)
            else:
                col_name = rule.name
                temp = np.zeros(df.shape[0], dtype=bool)
                for quantile in rule:
                    temp = np.logical_or(
                        (df.loc[:, col_name] == quantile).values, temp)
                    
                dates = df.period_start_date[temp].values
                hours = df.hour[temp].values
                upgraded_points += len(dates)
                
                for (i, hour) in enumerate(np.unique(hours)):
                    columns = [ch(c1, c2, hour)]
                    indices = dates[hours == hour]
                    res.loc[indices, columns] = dfs['Flin'].loc[indices, columns]
    return res

def compute_Fcmb(Flin, Flsq, rules, dates):
    info_dfs, rules = rules
    dfs = {"Flin" : Flin, "Flsq" : Flsq}
    select_dates(dfs, dates)
    order, edges = compute_order(dfs["Flin"])
    sort_dfs(dfs, order)    
    
    # Apply combination rules
    Fcmb = apply_rules(dfs, info_dfs, rules)  
    return Fcmb

##### Compute Funi
def compute_billateral(flux, order, fractions=[0, 0.1, 0.5, 0.75],
                       colors=["c", "b", "m", "r"]):
    x = 21
    y = 3
    data = np.zeros((x * y, 2, len(fractions)))
    for j in range(y):
        for i in range(x):
            z = j * x + i
            f = order[0:-1:24][z]
            
            c1 = f.split("_")[1]
            c2 = f.split("_")[3]

            flux_from=flux.loc[:,[f"{ch(c1, c2, h)}"
                                  for h in range(24)]].values.reshape(-1, 1).ravel()
            flux_to = flux.loc[:,[f"{ch(c2, c1, h)}"
                                  for h in range(24)]].values.reshape(-1, 1).ravel()
            
            for k, (frac, color) in enumerate(zip(fractions[::-1],colors[::-1])):
                pos_flux_from = np.where(flux_from > 0)[0]
                lower_flux_to=flux_to[pos_flux_from]<=flux_from[pos_flux_from]*frac
                if len(lower_flux_to) == 0:
                    value = 0
                else:
                    value = 100 * lower_flux_to.mean()
                    
                data[z, 0, k] = value
                
            for k, (frac, color) in enumerate(zip(fractions[::-1], colors[::-1])):
                pos_flux_from = np.where(flux_to > 0)[0]
                lower_flux_to=flux_from[pos_flux_from]<=flux_to[pos_flux_from]*frac
                if len(lower_flux_to) == 0:
                    value = 0
                else:
                    value = - 100 * lower_flux_to.mean()

                data[z, 1, k] = -value                    

    return data

def one_sided_flows(F, Fcmb, dates):
    dfs = {"Fcmb" : Fcmb, "F" : F}
    select_dates(dfs, dates)
    order, edges = compute_order(dfs["F"], s="DAFlux_")
    sort_dfs(dfs, order)

    # Compute the number of time where the flows are one-sided
    data = compute_billateral(dfs['F'], order)
    df_data = pandas.DataFrame(
        index=[e[0] + ' <> ' + e[1] for e in edges[0:63]],
        columns=["A -> B", "B -> A"],
        data=data[:, :, 1])

    # Select connections with more than 75% one-sideness
    unilateral = edges[0:63][np.where(np.min(data[:, :, 1], axis=1) > 75)[0]]
    return unilateral

def set_unilateral_to_zero(df, unilateral):
    res = copy.deepcopy(df)
    for c1, c2 in unilateral:
        columns_from = [ch(c1, c2, h) for h in range(24)]
        columns_to = [ch(c2, c1, h) for h in range(24)]
        for column_from, column_to in zip(columns_from, columns_to):
            export_id = np.where(df.loc[:, column_from]> df.loc[:, column_to])[0]
            export_dates = res.index[export_id]
            res.loc[export_dates, column_from] -=res.loc[export_dates, column_to]
            res.loc[export_dates, column_to] = 0

            import_id = np.where(df.loc[:, column_to]>=df.loc[:, column_from])[0]
            import_dates = res.index[import_id]
            res.loc[import_dates, column_to] -=res.loc[import_dates, column_from]
            res.loc[import_dates, column_from] = 0
            
    return res

def compute_Funi(Fcmb, os_flows):    
    Funi = set_unilateral_to_zero(Fcmb, os_flows)
    return Funi

################ Compute on test set and save results
def compute_test_set(F, rules, os_flows):
    datesh, data, model, network, dims = load_data("test")
    Pz, Pz_prime = load_prices(data, network, model, dims)
    Cz, Rz, Gz = load_fundamentals(data, network, model, dims)
    Vz = load_maximal_generation_capacities(datesh, network)
    Az_zprime = load_network_constraints(data, network, model, dims)
    S, M, b, B = formulate_lin_problem(Pz, Pz_prime, Cz, Rz, Vz, Az_zprime,
                                       network, model, dims)
    Flin = compute_Flin(S, M, b, B, network, (2, dims[1], dims[2]), datesh, data)
    
    f, x0, CD, B, b = formulate_lsq_problem(Pz, Pz_prime, Cz, Rz, Gz, Vz, Az_zprime,
                                            network, model, dims)
    Flsq = compute_Flsq(f,x0,CD,B, b, network, (2, dims[1], dims[2]), datesh, data)
    
    dfs = {"Flin" : Flin, "Flsq" : Flsq, "F" : F}
    select_dates(dfs, data[0])
    order, edges = compute_order(dfs["Flin"])
    sort_dfs(dfs, order)

    # Compute the quantiles of the fundamentals
    fundamentals = {
        "conso" : load("consumption"),
        "renewable_prod" : load("renewable_generation"),
        "programmable_prod" : load("programmable_generation")
    }
    select_dates(fundamentals, data[0])

    # Recompute the quantiles but keep the rules
    info_dfs, rules = rules
    info_dfs = {}
    for c1, c2 in edges:
        info_dfs[ch(c1, c2)] = get_edge_df(c1, c2, dfs, fundamentals, edges)

    rules = (info_dfs, rules)
    Fcmb = compute_Fcmb(Flin, Flsq, rules, data[0])
    Funi = compute_Funi(Fcmb, os_flows)

    return Az_zprime, Flin, Flsq, Fcmb, Funi, data[0]

################ Evaluation
def compare_flows(Flin, Flsq, F, c1, c2, dfkeys, dates, dates_=None, params={}):
    dfs = {"Flin" : Flin, "Flsq" : Flsq, "F" : F}
    select_dates(dfs, dates)    
    order, edges = compute_order(dfs["Flin"])
    sort_dfs(dfs, order)
    
    dfcolors = {"Flin" : "b", "Flsq" : "g", "F" : "r"}    
    with matplotlib.rc_context(
            { "text.usetex" : True,
              "text.latex.preamble" : r"\usepackage[bitstream-charter]{mathdesign} \usepackage[T1]{fontenc}",
              "font.family" : ""}):    
        fig, ax = plt.subplots(figsize=(19.2, 10.8))
        kref = list(dfs.keys())[0]
        if dates_ is None:
            dates_comp = [dfs[kref].index[0].date(), dfs[kref].index[-1].date()]
        else:
            dates_comp = dates_
        dates = [datetime.datetime.fromordinal(d.toordinal())
                 for d in dfs[kref].index
                 if ((d > dates_comp[0]) and (d < dates_comp[-1]))]
        datetimes = np.array(
            [d + datetime.timedelta(hours=h) for d in dates for h in range(24)])
        mins = []
        maxs = []
        for key in dfkeys:
            df = dfs[key]
            color = dfcolors[key]
            flux_from = df.loc[dates, [f"{ch(c1, c2, h)}"
                                       for h in range(24)]].values.reshape(-1, 1)
            flux_from = flux_from.ravel()
            mins += [np.min(flux_from)]
            maxs += [np.max(flux_from)]
            ax.plot(datetimes, flux_from, label=key, c=color,
                    linewidth=params["linewidth"])
    
        x_major_locator = mdates.DayLocator()
        ax.xaxis.set_major_locator(x_major_locator)        
        ax.tick_params(axis="both", labelsize=params["fontsize_labels"])

        y_major_formatter = tck.FuncFormatter(
            lambda x, pos: round(x/1000, ndigits=1))
        ax.yaxis.set_major_formatter(y_major_formatter)
        
        ax.grid("on", axis="x",which="major", linestyle="-", linewidth=1, color="k")
        ax.grid("on", axis="y",which="major", linestyle="-", linewidth=1, color="k")
        
        ax.set_xlabel("Time", fontsize=params["fontsize"])
        ax.set_ylabel("\\textbf{F} (GWh)", fontsize=params["fontsize"])
        ax.set_title(f"From {latex_zone(c1)} to {latex_zone(c2)}",
                     fontsize=params["fontsize"])
        ax.set_xlim([dates[0], dates[-1]])
        ax.set_ylim([min(mins) - 25, max(maxs)])    
        ax.legend(loc="upper right", fontsize=params["fontsize"])
        plt.show()

def compute_errors_test(A, F, Flin, Flsq, Fcmb, Funi, dates):
    dfs={"A":A, "Flin" : Flin, "Flsq" : Flsq, "F" : F, "Fcmb" : Fcmb,"Funi" : Funi}
    select_dates(dfs, dates)

    # Compute order, sort, reshape
    order, edges = compute_order(dfs["Flin"])
    sort_dfs(dfs, order)
    matrices = matrixize(dfs)

    metrics = [CC, mae, smape]    
    yhats = [matrices[k] for k in list(matrices.keys()) if k != "F"]
    ytrue = matrices["F"]
    res = np.zeros((len(metrics), len(yhats), ytrue.shape[1]))
    for i, m in enumerate(metrics):
        for j, yhat in enumerate(yhats):
            res[i, j] = m(ytrue, yhat, mean=False)

    res_mean = pandas.DataFrame(
        index = [k for k in matrices.keys() if k != "F"],
        columns = [m.__name__ for m in metrics],
        data = res.mean(axis=2).transpose())
    return res_mean

def compute_DM_tests(A, F, Flin_test, Flsq_test, Fcmb_test, Funi_test, dates):
    dfs={"A":A, "Flin" : Flin_test, "Flsq" : Flsq_test, "F" : F,
         "Fcmb" : Fcmb_test,"Funi" : Funi_test}
    select_dates(dfs, dates)
    order, edges = compute_order(dfs["Flin"])
    sort_dfs(dfs, order)
    
    Y = dfs["F"].values
    models = [k for k in list(dfs.keys()) if k != "F"]
    nm = len(models)

    pvalues_smape = np.zeros((nm, nm))
    for i in range(nm):
        for j in range(nm):
            m1 = models[i]
            m2 = models[j]
            if i == j:
                pvalues_smape[i, j] = np.nan         
            else:
                pvalues_smape[i, j]=DM(Y, dfs[m1].values, dfs[m2].values)
    
    df_smape = pandas.DataFrame(columns=models, index=models,data=pvalues_smape)
    return df_smape

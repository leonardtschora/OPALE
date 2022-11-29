import os, copy, joblib, pandas, datetime, numpy as np, time
from collections import OrderedDict
from sklearn.metrics import mean_absolute_error

import src.models.parallel_scikit as ps
import src.models.model_utils as mu
from src.analysis.metrics import mae, smape, CC
from src.models.data_scaler import DataScaler

class ModelWrapper(object):
    """
    Model wrapper around models. Facilitates the workflows around
    models in general and enables parallel grid searches around models.
    Enables using the oob's predictions for evaluating a model's quality during 
    grid searches.
    """
    def __init__(self, prefix, dataset_name, country="", spliter=None,
                 predict_two_days=False, known_countries=[], flow_estimation="",
                 countries_to_predict="all"):
        self.prefix = prefix
        self.dataset_name = dataset_name
        self.country = country

        # Spliter to give to all components of the model
        self.spliter = spliter
        self.external_spliter = spliter
        self.validation_mode = "external"
        
        self.columns = self.load_columns()
        self.date_cols = self.load_date_columns()
        self.label = self.load_labels()
        self.past_label = [f"{l}_past_1" for l in self.label]

        self.predict_two_days = predict_two_days
        self.turning_hour = 9 # We take all prices before 9am
        
        # Specify if we load graph data or not (for transforming edges)
        self.known_countries = known_countries
        self.countries_to_predict_ = countries_to_predict
        self.replace_ATC = flow_estimation

        # For differentiating price and edge labels,
        # will be overwritten for graphs
        self.nodes_per_country = 1
        self.n_out_per_nodes = 24
        

    def string(self):
        return "ModelWrapper"

    def shuffle_train(self):
        return True
        
    def save_name(self):
        return mu.save_name(self.prefix, self.dataset_name)

    def save_path(self):
        return mu.save_path(self.prefix, self.dataset_name)
    
    def folder(self):
        folder = mu.folder(self.dataset_name)
        if not os.path.exists(folder): os.mkdir(folder)
        return folder

    def results_path(self):
        path = os.path.join(os.environ["OPALE"], "data", "Grid Search")
        name = self.string() + f"_results.csv"
        return os.path.join(path, name)        
    
    def train_dataset_path(self):
        return mu.train_dataset_path(self.dataset_name)

    def test_dataset_path(self):
        return mu.test_dataset_path(self.dataset_name)

    def all_dataset_path(self):
        return mu.all_dataset_path(self.dataset_name)

    def extra_dataset_path(self):
        return mu.extra_dataset_path(self.dataset_name)        

    def figure_folder(self):
        return mu.figure_folder(self.dataset_name)

    def model_path(self):
        return mu.model_path(self.prefix, self.dataset_name)

    def train_prediction_path(self):
        return mu.train_prediction_path(self.prefix, self.dataset_name)

    def test_prediction_path(self):
        return mu.test_prediction_path(self.prefix, self.dataset_name)

    def val_prediction_path(self):
        return mu.val_prediction_path(self.prefix, self.dataset_name)
    
    def validation_prediction_path(self, i):
        bp = self.val_prediction_path()
        return bp[:-4] + f"_{str(i)}.csv"

    def tr_prediction_path(self, i):
        # Does not supports filters
        bp = self.test_recalibrated_prediction_path()
        return bp[:-4] + f"_{str(i)}.npy"

    def tr_shap_path(self, i):
        bp = self.test_recalibrated_shape_path()
        return bp[:-4] + f"_{str(i)}.npy"
    
    def test_recalibrated_prediction_path(self, filters=None,inverted_filters=None):
        return mu.test_recalibrated_prediction_path(self.string(), self.replace_ATC)
    
    def extra_prediction_path(self):
        return mu.extra_prediction_path(self.prefix, self.dataset_name)

    def all_prediction_path(self):
        return mu.all_prediction_path(self.prefix, self.dataset_name)

    def test_shape_path(self):
        return mu.test_shape_path(self.prefix, self.dataset_name)

    def test_recalibrated_shape_path(self):
        return mu.test_recalibrated_shape_path(self.string(), self.replace_ATC)

    def _params(self, ptemp):
        p = self.params()
        p.update(ptemp)
        return p

    def get_scaler(self, ptemp):
        try:
            scaler = DataScaler(ptemp["scaler"], spliter=self.spliter)
        except:
            scaler = DataScaler("", spliter=self.spliter)
            
        try: del ptemp["scaler"]
        except: pass
        
        return scaler

    def get_transformer(self, ptemp):
        try:
            transformer = DataScaler(ptemp["transformer"], spliter=self.spliter)
        except:
            transformer = DataScaler("", spliter=self.spliter)
            
        try: del ptemp["transformer"]
        except: pass
        
        return transformer

    def get_search_space(self, fast=False, country=None, version=None, n=None,
                         stop_after=-1):
        pass        
    
    def params(self):
        return {}
    
    def make(self, ptemp):
        pass

    def prepare_for_make(self, ptemp, GNN="drop_date"):
        ptemp_ = copy.deepcopy(ptemp)
        if "seeds" in ptemp_:
            del ptemp_["seeds"]
            
        if self.countries_to_predict_ == "not_graph":            
            scaler = self.get_scaler(ptemp_)
            transformer = self.get_transformer(ptemp_)
        else:
            node_scaler = self.get_scaler(ptemp_)
            scaler = DataScaler(node_scaler.scaling, psd_idx=self.psd_idx,
                                country_idx=self.country_idx,
                                edges_columns_idx=self.edges_columns_idx,
                                GNN=GNN)
            transformer = self.get_transformer(ptemp_)
        return scaler, transformer, ptemp_

    def predict(self, regr, X):
        return regr.predict(X)
    
    def eval(self, regr, X, y): 
        yhat = self.predict(regr, X)
        return mean_absolute_error(y, yhat)       
    
    def predict_val(self, regr, X, oob=False):
        return self.predict(regr, X)
    
    def eval_val(self, regr, X, y, oob=False):
        return self.eval(regr, X, y)

    def predict_test(self, regr, X):
        return self.predict_test_(regr, X)
    
    def predict_test_(self, regr, X):
        ypred = self.predict(regr, X)
        if not self.predict_two_days:
            return ypred
        else:
            y_past = X[:, self.past_label_idx]
            ypred = np.concatenate(
                (y_past[:, self.turning_hour:], ypred,
                 y_past[:, :self.turning_hour]),
                axis=1
            )
            return ypred
            
    def eval_test(self, regr, X, y):
        return self.eval(regr, X, y)        
        
    def save(self, model):
        joblib.dump(model, self.model_path())

    def load(self):
        return joblib.load(self.model_path())

    def load_dataset(self, path):
        if self.countries_to_predict_ == "not_graph":
            dataset, labels, past_label_idx =  mu.load_dataset(
                path, self.label, self.past_label)
            self.past_label_idx = past_label_idx
            return mu.load_dataset_(dataset, labels, self.label)
        else:
            return self.load_graph_dataset(path)

    def update_labels(self):
        """
        Update the labels given the list of known countries and the countries to
        predict.
        Assume everything is sorted
        """
        if self.countries_to_predict_ == "all":
            country_list = np.unique(
                np.array([c.split("_")[0] for c in self.label]))
        else:
            country_list = self.countries_to_predict_
            
        self.countries_to_predict = [c for c in set(country_list).difference(
            self.known_countries)]
        self.label = [f"{c}_price_{h}" for c in self.countries_to_predict
                      for h in range(24)]

    def update_columns(self):
        self.columns = np.array(self.columns)
        for c in self.known_countries:
            for h in range(24):
                past_price = f"{c}_price_{h}_past_1"
                price = f"{c}_price_{h}"
                self.columns[np.where(self.columns == past_price)] = price
                
        
    def load_graph_dataset(self, path):
        self.edges_columns = self.load_atc_columns()
        self.update_labels()
        self.update_columns()        
        
        # Override loading behavior : this will return X as a dataframe        
        # with the date column (for the splitter).
        dataset = pandas.read_csv(path)                
        labels = dataset[np.array(self.label)]
        dataset.drop(columns=self.label, inplace=True)        
        self.psd_idx = np.where(dataset.columns == "period_start_date")[0][0]
        
        # Sort the names        
        dataset = dataset[np.concatenate((np.array(["period_start_date"]),
                                          np.array(self.columns)))]
        
        # Compute the country indices in order to sort by node
        columns_country = np.array([c.split("_")[0] for c in dataset.columns
                                    if c not in self.edges_columns])
        _, idxs = np.unique(columns_country, return_index=True)
        country_list = np.array([c for c in columns_country[np.sort(idxs)]
                                 if c != "period"])
         # Get country code of the column if it is not an edge column
        columns_mask = np.array([c.split("_")[0] if c not in np.concatenate(
            (self.edges_columns, ["period_start_date"])) else False
                        for c in dataset.columns])
         # Compute the indices of the data of each country in the dataset
        country_idx = OrderedDict()
        for country in country_list:
            country_idx[country] = np.where(columns_mask == country)[0]        
        self.country_idx = copy.deepcopy(country_idx)
         # Compute the edge indices (of each countries)
        country_idx = OrderedDict()
        def rebuild(ss, car="_"):
            s = ""
            for s_ in ss: s += s_ + car
            return s[:-len(car)]
        
        atc_columns = np.array(
            [rebuild(c.split("_")[:5]) if c
             in self.edges_columns else False for c in dataset.columns])        
        for c1 in country_list:
            for c2 in country_list:
                prefix = f"ATC_from_{c1}_to_{c2}"
                key = f"{c1}_{c2}"
                country_idx[key] = []
                if len(self.edges_columns) != 0:
                    idx_in_edge_columns = np.where(
                        atc_columns == np.array(prefix))[0]
                    if len(idx_in_edge_columns) > 0:
                        country_idx[key] = idx_in_edge_columns
                         
        self.edges_columns_idx = country_idx
        
        # Sort labels in the same order as the data
        # Compute sorting indices
        sort_label_idx = []
        n_out_prices = self.n_out_per_nodes * len(self.countries_to_predict) * self.nodes_per_country
        price_labels = np.array(self.label[:n_out_prices])
        for l in price_labels:
            prefix = l.split("_")[0]
            if prefix != "Flux":
                sort_label_idx.append(
                    np.where(np.array(country_list) == prefix)[0][0])

        sort_label_idx = np.array(sort_label_idx)
        
        sort_indices = np.zeros(n_out_prices, dtype=int)
        unique_indices = np.unique(sort_label_idx)
        current = np.sort(unique_indices)[0]
        for i in np.sort(unique_indices):
            inds = np.where(sort_label_idx == i)[0]
            ninds = len(inds)
            sort_indices[current:current+ninds] = inds
            current += ninds
        price_labels = np.array(price_labels)[np.array(sort_indices)]

        # Sort the edge labels
        edge_labels = np.array(self.label[n_out_prices:])
        edge_idx = []
        if len(edge_labels) != 0:
            for i, c1 in enumerate(list(self.country_idx.keys())):
                for j, c2 in enumerate(list(self.country_idx.keys())):
                    for h in range(24):
                        name = f"Flux_from_{c1}_to_{c2}_{h}"
                        index = np.where(np.array(edge_labels) == str(name))[0]
                        if len(index) != 0:
                            edge_idx += list(index)
                        
        if len(edge_idx) > 0:
            edge_labels = np.array(edge_labels)[np.array(edge_idx)]
        else:
            edge_labels = np.array([])
        
        self.label = np.concatenate((price_labels, edge_labels))

        # Replace the atcs if specified
        if self.replace_ATC != "":
            ATC_file = os.path.join(os.environ["OPALE"], "data", "Optim",
                                    f"{self.replace_ATC}.csv")        
            ATCs = pandas.read_csv(ATC_file, index_col="period_start_date")
            ATCs = ATCs.loc[dataset.period_start_date]
            for edge in self.edges_columns:
                dataset.loc[:, edge] = np.nan_to_num(ATCs.loc[:, edge], nan=0)
            
         # RECOMPUTE countries_to_predict using the correct sorting
        self.countries_to_predict = [c for c in self.country_idx.keys()
                                     if c in self.countries_to_predict]
        self.past_label = [f"{l}_past_1" for l in self.label]                
        self.past_label_idx = mu.past_label_columns(
            dataset, self.past_label)
        X = dataset.values
        y = labels[self.label].values
        
        return X, y        
    
    def load_train_dataset(self):
        return self.load_dataset(self.train_dataset_path())
    
    def load_test_dataset(self):
        return self.load_dataset(self.test_dataset_path())

    def load_columns(self, order_str=""):
        return mu.load_columns(self.dataset_name, order_str=order_str)

    def load_date_columns(self):
        return mu.load_columns(self.dataset_name, order_str="date")

    def load_atc_columns(self):
        return mu.load_columns(self.dataset_name, order_str="atc")        

    def load_labels(self):
        return mu.load_labels(self.dataset_name)
    
    def load_prediction_dataset(self):
        dataset = pandas.read_csv(self.test_dataset_path())
        dataset.drop(columns=["period_start_date"], inplace=True)
        X = dataset.values
        return X

    def load_base_results(self):
        p1 = os.path.split(mu.folder(self.dataset_name))[0]
        p2 = os.path.split(self.dataset_name)[0]
        p3 = self.prefix
        path = os.path.join(p1, f"{p3}_{p2}_results.csv")
        return self.load_results(path=path)
        
    def map_dict(self):
        return {"spliter" : (mu.spliter_to_string, mu.spliter_from_string)}

    def load_results(self, path=None):
        if path is None: path = self.results_path()        
        map_dict = self.map_dict()
        return mu.load_results(path, map_dict)

    def filter_results(self, df, filters, inverted_filters):
        df.loc[:, "file"] = copy.deepcopy(df.index)
        if filters != {} or inverted_filters != {}:
            indices = np.ones(df.shape[0], dtype=bool)
            which = np.concatenate((
                np.ones(len(list(filters.keys()))),
                np.zeros(len(list(inverted_filters.keys())))))            
            for w, col in zip(
                    which, list(filters.keys()) + list(inverted_filters.keys())):
                if w == 1:
                    indices = np.where(df.loc[indices, col] == filters[col])[0]
                if w == 0:
                    indices = np.where(
                        df.loc[indices, col] != inverted_filters[col])[0]

            if len(indices) < 0:
                raise Exception("Filters found no valid configuration!")
            else:
                df = df.loc[indices]
                df.index = np.arange(df.shape[0])

        return df        
    
    def compute_accs(self, df, recompute=True):
        # Handle the spliter : it needs not to shuffle data for retrieving Yv!!
        try:
            stemp = self.spliter.shuffle
            self.spliter.shuffle = False
        except:
            pass

        try:
            estemp = self.external_spliter.shuffle
            self.external_spliter.shuffle = False
        except:
            pass
                
        maes, smapes, accs = self.compute_metrics(df, recompute=recompute)

        try:
            self.spliter.shuffle = stemp
        except:
            pass

        try:                
            self.external_spliter.shuffle = estemp
        except:
            pass

        return accs

    def recompute_accs(self):
        # Recompute all accs once and for all
        df = self.load_results()
        df.loc[:, "file"] = df.index
        accs = self.compute_accs(df)
        df.loc[:, "accs"] = accs

        l = []
        results = []
        seeds = []
        for i in range(df.shape[0]):
            d = dict(df.iloc[i])
            seeds += [d.pop("seeds")]
            results += [(d.pop("maes"), d.pop("accs"), d.pop("times"))]
            l += [d]
            
        df = ps.results_to_df(results, l, seeds=None,
                              map_dict=self.map_dict())
        df.drop(columns="file", inplace=True)
        df.to_csv(self.results_path(), index=False)

    def best_params(self, df, for_recalibration=False, acc=False,
                    filters={}, inverted_filters={}, recompute=True):        
        df = self.filter_results(copy.deepcopy(df), filters, inverted_filters)
        
        if not acc:
            best_row = df.maes.argmin()
            best_params = df.loc[best_row].to_dict()
            print(f"BEST MAE = {round(best_params['maes'], ndigits=2)}")            
        else:
            df.loc[:, "accs"] = self.compute_accs(df, recompute=recompute)
            best_row = df.accs.argmax()
            best_params = df.loc[best_row].to_dict()
            print(f"BEST ACC = {round(best_params['accs'], ndigits=3)}")

        best_params.pop("file")            
        best_params.pop("times")            
        params = self.params()        
        params.update(best_params)
        if for_recalibration:
            if "stop_after" in params.keys():
                params["stop_after"] = -1
            params.pop("maes")
            if acc:
                params.pop("acc")
                   
        return params
        
    def save_preds(self, regr, trades_col):
        X, y = self.load_train_dataset()        
        Xt, yt = self.load_test_dataset()
        self.save_train_preds(regr, X, trades_col)
        self.save_test_preds(regr, Xt, trades_col)       
        
    def save_train_preds(self, regr, X, trades_col):
        train_dataset = pandas.read_csv(self.train_dataset_path())
        train_prevs = pandas.DataFrame({"date_col" : train_dataset["date_col"].values,
                                        "predicted_prices" : self.predict(regr, X),
                                        "real_prices" : train_dataset[self.label],
                                        "trades" : train_dataset[trades_col]})
        train_prevs.to_csv(self.train_prediction_path(), index=False)

    def save_test_preds(self, regr, Xt, trades_col):
        test_dataset = pandas.read_csv(self.test_dataset_path())
        test_prevs = pandas.DataFrame({"date_col" : test_dataset["date_col"].values,
                                       "predicted_prices" : regr.predict(Xt),
                                       "real_prices" : test_dataset[self.label],
                                       "trades" : test_dataset[trades_col]})
        test_prevs.to_csv(self.test_prediction_path(), index=False)     

    def test_and_save(self, best_params, trades_col, trades_dataset,
                      cv=None, same_val=False, oob=False):
        X, y = self.load_train_dataset()        
        Xt, yt = self.load_test_dataset()

        for k in self.map_dict().keys():
            v = self.map_dict()[k]
            best_params[k] = v[1](best_params[k]) 
        
        regr = self.make(self._params(best_params))
        regr.fit(X, y)               

        self.save(regr)

        print("Best model saved at " + self.model_path())
        print("TEST SET MAE = " + str(mu.mae_epsilon(yt, regr.predict(Xt))))
        print("TEST SET SMAPE = " + str(mu.smape_epsilon(yt, regr.predict(Xt))))
        
        ##### Trade metrics
        test_trades_path = mu.test_dataset_path(trades_dataset)
        test_trades = pandas.read_csv(test_trades_path)[trades_col].values
        print("TEST SET GAIN = " + str(mu.gain(yt, regr.predict(Xt), test_trades)))
        print("TEST SET DIRECTION = " + str(mu.direction(yt, regr.predict(Xt), test_trades)))
        return regr

    def set_jobs(self, regr, njobs):
        pass

    def test_and_save_epf(self, best_params, njobs=1, save_model=True,
                          save_preds=True):
        regr = self.make(self._params(best_params))
        self.set_jobs(regr, njobs)
        oob = self.validation_mode == "oob"
        
        # Recompute validation error
        X, y = self.load_train_dataset()        
        X, y, Xv, yv = ps.outer_validation(
            self.validation_mode, self.external_spliter, X, y)
        Xv, yv = ps.inner_validation(self.validation_mode, X, y, Xv, yv, self)
        pandas.DataFrame(yv).to_csv("validation.csv")
        
        try:
            seed = best_params["seeds"]
        except:
            seed = None
            
        if seed is not None:
            ps.set_all_seeds(seed)
            print(f"Using SEED {seed}")
            
        regr.fit(X, y)
        ypred = self.predict_val(regr, Xv, oob=oob)
        print(X.shape, Xv.shape)
        print("(RECOMPUTED) VAL SET MAE = " + str(round(mae(yv, ypred), ndigits=2)))
        print("(RECOMPUTED) VAL SET SMAPE = " + str(round(smape(yv, ypred),
                                                          ndigits=2)))
        val_prevs = pandas.DataFrame(ypred)
        if save_preds: val_prevs.to_csv(self.val_prediction_path(), index=False)

        # Compute test set error
        X, y = self.load_train_dataset()
        Xt, yt = self.load_test_dataset()
        
        if seed is not None: ps.set_all_seeds(seed)
        regr.fit(X, y)          
        
        ypred = self.predict_test(regr, Xt)
        test_prevs = pandas.DataFrame(ypred)
        if save_preds:
            test_prevs.to_csv(self.test_prediction_path(), index=False)
        print("TEST SET MAE = " + str(round(mae(yt, ypred), ndigits=2)))
        print("TEST SET SMAPE = " + str(round(smape(yt, ypred), ndigits=2)))

        if save_model:
            self.save(regr)
            print("Best model saved at " + self.model_path())            
        return regr

    def load_recalibrated_predictions(self, start, stop, step):
        results = np.zeros((stop - start, len(self.label)))
        for i in range(start, stop, step):
            sb = start+i
            se = start+i+step
            print(sb, se)
            results[sb:se, :] = np.load(
                self.tr_prediction_path(start + i))
        return results

    def recalibrate_epf(self, regr=None, best_params={}, ncpus=1, seed=None,
                        start=None, stop=None, step=1, calibration_window=None,
                        save=True, n_shap=0, filters=None, inverted_filters=None):
        if self.spliter is not None and self.spliter.shuffle:
            print("The validation set (early stopping) will be randomly picked")

        # Load data
        Xtrain, ytrain = self.load_train_dataset()
        Xt, yt = self.load_test_dataset()

        # Handle start and stop if unspecified
        if start is None: start = 0
        if stop is None: stop = yt.shape[0]
        
        # Select the data between start and stop
        Xtrain = np.concatenate((Xtrain, Xt[:start]))
        ytrain = np.concatenate((ytrain, yt[:start]))            
        Xt = Xt[start:stop]
        yt = yt[start:stop]
        ntest = yt.shape[0]
        
        # Paralellize the computations
        predictions = np.zeros_like(yt)
        times = np.zeros(len(range(0, ntest, step)))
        
        try:
            n_labels = yt.shape[1]
        except:
            n_labels = 1
        if n_shap > 0:
            shaps = np.zeros((n_labels, yt.shape[0], Xt.shape[1]), dtype=np.float16)
        else:
            shaps = None
            
        print(f"RECALIBRATING {stop -start} DAYS ON {calibration_window} SAMPLES")
        
        self.perform_recalibration(
            predictions, best_params, Xtrain, ytrain, Xt, yt, seed, times, ncpus,
            ntest, start, shaps=shaps, calibration_window=calibration_window,
            n_shap=n_shap, step=step, save=save)

        # Compute metrics
        print("RECALIBRATED TEST SET MAE = ",
              str(round(mae(yt, predictions), ndigits=2)))
        print("RECALIBRATED TEST SET SMAPE = ",
              str(round(smape(yt, predictions), ndigits=2)))
        total_time = np.sum(times)
        print(f"RECALIBRATED TEST TIME = " + str(total_time))

        # Save recalibrated predictions
        test_prevs = pandas.DataFrame(predictions)
        if n_shap == 0:
            test_prevs.to_csv(
                self.test_recalibrated_prediction_path(
                    filters=filters,
                    inverted_filters=inverted_filters),
                index=False)
        if n_shap > 0: np.save(self.test_recalibrated_shape_path(), shaps)
        return total_time

    def perform_recalibration(self, predictions, best_params, Xtrain, ytrain,
                              Xt, yt, seed, times, ncpus, ntest, start, shaps=None,
                              calibration_window=None, n_shap=0, step=1,save=False):
        ps.recalibrate_parallel(
            predictions, self, best_params, Xtrain, ytrain, Xt, yt, seed, times,
            ncpus, ntest, start, shaps=shaps, calibration_window=calibration_window,
            n_shap=n_shap, step=step, save=save)

    def get_country_code(self):
        if self.country == "NP":
            code = "NOR"
        elif self.country == "PJM":
            code = "US"
        else:
            code = self.country
        return code

    def load_val_pred(self, i):
        path = self.validation_prediction_path(i)
        df = pandas.read_csv(path)
        return df

    def load_best_val(self):
        df = self.load_results()
        i = df.sort_values(by="maes").head(1).index.values[0]
        print(self.string(), i)
        return self.load_val_pred(i)
    
    def load_all_validation_results(self, df,n_combis=None, n_val=None, start=None):
        if n_combis is None:
            n_combis = self.load_results().shape[0]
        if n_val is None:
            n_val = self.spliter.validation_ratio
        if start is None:
            start=0
            
        dfs = np.zeros(
            (n_combis, n_val, len(self.countries_to_predict), 24))
        success = []    
        for i in range(n_combis):
            try:
                f = df.loc[i+start, "file"]
                res = self.load_val_pred(f)
                dfs[i] = res.values.reshape(n_val,len(self.countries_to_predict),
                                           24)
                success.append(i)
            except:
                pass
                #print(f"{f} FAILED")
        return dfs, success
    
    def compute_metrics(self, df, seg=25, recompute=True):
        if not recompute:
            if "accs" in df.columns:
                return None, None, df.accs.values
            
        n_combis = df.shape[0]
        
        nz = len(self.countries_to_predict)
        X, y = self.load_train_dataset()
        X, y, Xv, yv = ps.outer_validation(
            self.validation_mode, self.external_spliter, X, y)
        Xv, yv = ps.inner_validation(self.validation_mode, X, y, Xv, yv, self)  
        yv = yv.reshape(yv.shape[0], nz, 24)

        Maes = np.zeros((n_combis, nz))
        Smapes = np.zeros((n_combis, nz)) 
        Accs = np.zeros(n_combis)      
        for i in range(0, n_combis, seg):
            #print(i, i+seg)
            dfs, success = self.load_all_validation_results(
                df, n_combis=seg, start=i)
            seg_ = min(seg, n_combis - i)
            dfs = dfs[:seg_]
            
            abs_diff = np.abs(dfs - yv)
            Maes[i:i+seg_] = abs_diff.mean(axis=(1, 3))
            Smapes[i:i+seg_]=(200*abs_diff / (np.abs(dfs) + np.abs(yv))).mean(axis=(1, 3))

            yhat_mean = dfs.mean(axis=(1, 3))
            y_mean = yv.mean(axis=(0, 2))

            yhat_centered = np.zeros_like(dfs)
            yv_centered = np.zeros_like(yv)
            for j in range(dfs.shape[-1]):
                yv_centered[:, :, j] = yv[:, :, j] - y_mean
                for k in range(dfs.shape[1]):
                    yhat_centered[:, k, :, j] = dfs[:, k, :, j] - yhat_mean
                
            num = (yhat_centered * yv_centered).sum(axis=(1, 3))
            d1 = (yhat_centered * yhat_centered).sum(axis=(1, 3))
            d2 = (yv_centered * yv_centered).sum(axis=(0, 2))
            Accs[i:i+seg_] = (num / np.sqrt(d1 * d2)).mean(axis=1)
        
        return Maes, Smapes, Accs

    def plot_errors(self, success):
        maes, smapes, rmaes, accs, success= self.compute_metrics(
            self.load_results())
        
        plt.plot(np.clip(maes[success], 0, 15).transpose())
        plt.plot(np.clip(smapes[success], 0, 30).transpose())

        plt.plot(np.clip(maes[success], 0, 15).mean(axis=1), label="maes")
        plt.plot(np.clip(smapes[success], 0, 30).mean(axis=1), label="smapes")
        plt.plot(accs, label="accs")

    def get_country_errors(self, dfs, countries=["FR", "DE", "BE"]):
        indices = np.zeros(len(countries), dtype=int)
        for i, c in enumerate(countries):
            indices[i] = np.where(np.array(self.countries_to_predict) == c)[0][0]
        print(indices)
        
        maes, smapes, rmaes, accs = self.compute_metrics(dfs)
        res = np.zeros((len(countries), len(success), 3))        
        res[:, :, 0] =  maes[np.array(success)][:, indices].transpose()
        res[:, :, 1] =  smapes[np.array(success)][:, indices].transpose()
        res[:, :, 2] =  rmaes[np.array(success)][:, indices].transpose()
        return res    

    def replace_ATC_string(self, F=False):
        if F: res = "\_F"
        else: res = "\_"
        if self.replace_ATC == "":
            if "NEConv" in self.prefix:
                return res + "NE"
            else:
                return "\_A"
        if self.replace_ATC == "WithPrice":
            return res + "lin"
        if self.replace_ATC == "LeastSquares":
            return res + "lsq"
        if self.replace_ATC == "combined":
            return res + "cmb"
        if self.replace_ATC == "combined_unilateral":
            return res + "os"      

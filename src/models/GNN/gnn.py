import pandas, torch, numpy as np, copy, os, pandas, datetime, time
from collections import OrderedDict
from datetime import date
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

from src.models.GNN.torch_gnn import TorchGNN
from src.models.GNN.callbacks import *
from src.models.model_utils import flatten_dict

class GraphNeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(self, name, model_):
        self.name = name
        self.model_ = model_
        self.connection_path = model_["connection_path"]
        self.matrix = pandas.read_csv(self.connection_path, index_col="Country")
        self.N_OUTPUT = model_["N_OUTPUT"]
        self.channels = model_["channels"]
        self.edge_channels = model_["edge_channels"]
        self.activation = model_["activation"]
        self.regr = model_["regr"]
        self.edge_regr = model_["edge_regr"]
        self.batch_norm = model_["batch_norm"]  
        
        self.n_epochs = int(model_["n_epochs"])
        self.batch_size = int(model_["batch_size"])
        
        self.optimizer = model_["optimizer"]
        self.learning_rate = model_["learning_rate"]
        self.criterion = model_["criterion"]
        self.shuffle_train = model_["shuffle_train"]

        self.adapt_lr = model_["adapt_lr"]
        self.early_stopping = model_["early_stopping"]
        self.early_stopping_alpha = model_["early_stopping_alpha"]
        self.early_stopping_patience = model_["early_stopping_patience"]        
        
        self.dropout = model_["dropout"]
        self.pooling = model_["pooling"]
        self.spliter = model_["spliter"]    

        self.country_idx = model_["country_idx"]
        self.edges_columns_idx = model_["edges_columns_idx"]    
        self.psd_idx = model_["psd_idx"]        
        self.country_list = [c for c in self.country_idx.keys()]

        self.is_dynamic = model_["is_dynamic"]
        self.edge_DEBE = model_["edge_DEBE"]
        self.split_DEAT = model_["split_DEAT"]
        self.layer_type = model_["layer_type"]        
        self.is_weighted = self.layer_type not in ("GINConv", "GCNConv")
        self.heads = model_["heads"]        

        self.n_out_per_nodes = model_["n_out_per_nodes"]
        self.n_out_per_edges = model_["n_out_per_edges"]
        self.known_countries = model_["known_countries"]
        self.countries_to_predict = model_["countries_to_predict"]

        # Nodes
        self.nodes_per_country = model_["nodes_per_country"]        
        self.n_nodes = len(self.country_list) * self.nodes_per_country
        self.n_out_nodes = int(len(self.countries_to_predict) * self.nodes_per_country)
        self.n_outputs_nodes = self.n_out_nodes * self.n_out_per_nodes

        # Edges
        self.n_out_edges = int(len(flatten_dict(self.edges_columns_idx)) / self.n_out_per_nodes)        
        if self.nodes_per_country == 1:
            n_edges = self.n_out_edges
        else:
            n_edges = self.n_out_edges + 2 * (self.nodes_per_country * len(self.country_list))
        self.n_edges = n_edges
        self.n_outputs_edges = self.n_out_edges * self. n_out_per_edges
            
        self.last_seen = None

        nz = len(self.known_countries) + len(self.countries_to_predict)
        train_mask = np.zeros(len(self.known_countries), dtype=int)
        for i, kc in enumerate(self.known_countries):
            idx = np.where(np.array(self.country_list) == kc)[0][0]
            train_mask[i] = idx

        self.test_mask =[i for i in range(nz) if i not in train_mask]
        self.test_mask = [self.nodes_per_country * i + h
                          for i in self.test_mask
                          for h in range(self.nodes_per_country)]        

    def shape_info(self):
        countries = f"There are {len(self.country_list)} countries and {len(self.countries_to_predict)} of them are to be predicted."
        nodes = f"There are {self.n_nodes} nodes ({self.nodes_per_country} per countries) so there are {self.n_out_nodes} output nodes resulting in {self.n_outputs_nodes} outputs for the nodes."
        edges = f"There are {self.n_edges} edges. {self.n_out_edges} are to be predicted resulting in {self.n_outputs_edges} outputs for the edges."
        return "\n" + countries + "\n" + nodes + "\n"+ edges
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():            
            setattr(self, parameter, value)
        return self    

    def update_params(self, input_shape=None, edge_shape=None):
        """
        Reset GNN's networks
        Reset this GNNRegressor params:
         (otpimizer, training state, callbacks, etc)
        To call everytime an hyperparameter changes!
        """

        try:
            self.edge_channels[0]
        except:
            if (self.edge_channels is None) or (np.isnan(self.edge_channels)):
                ec = None
            else:
                ec = int(self.edge_channels)
            self.edge_channels = [ec for c in self.channels]
        
        # Instantiate the model
        self.model = self.create_network(
            input_shape=input_shape, edge_shape=edge_shape)

        # Set callbacks
        self.callbacks = []
        self.early_stopping_callbacks()
        self.adapt_lr_callbacks()        

    def adapt_lr_callbacks(self):
        pass

    def early_stopping_callbacks(self):
        if self.early_stopping == "sliding_average":
            self.callbacks.append(
                EarlyStoppingSlidingAverage(
                    monitor="val_loss",
                    alpha=self.early_stopping_alpha,
                    patience=self.early_stopping_patience))          
    
    ###### METHODS FOR SKLEARN AND OPALE
    def create_network(self, input_shape=None, edge_shape=None):        
        return TorchGNN(
            self.batch_size, self.channels, self.activation, input_shape,
            self.n_outputs_nodes,  self.criterion, self.optimizer,
            self.learning_rate, self.layer_type, edge_shape,
            self.n_nodes, self.n_edges, self.n_out_per_nodes,            
            self.n_out_nodes, self.test_mask,
            heads=self.heads, regr=self.regr, batch_norm=self.batch_norm,
            lam=0.5, dropout=self.dropout, pooling=self.pooling,
            edge_channels=self.edge_channels)
    
    def fit(self, X, y, verbose=0, Xv=None, yv=None):
        # Prepare the data
        self.convert_indices(X)
        train_loader, val_loader = self.prepare_for_train(X, y, Xv=Xv, yv=yv)   
        
        # Create the GNN
        d = train_loader.dataset[0].x.shape[1]
        e1 = train_loader.dataset[0].edge_attr
        if e1 is None: e = None
        else: e = e1.shape[1]
        self.update_params(input_shape=d, edge_shape=e)
                
        # Create the trainer
        self.trainer = pl.Trainer(max_epochs=self.n_epochs,
                                  callbacks=self.callbacks,
                                  logger=True, enable_checkpointing=False,
                                  log_every_n_steps=1,
                                  enable_progress_bar=True)

        # Train
        self.trainer.fit(self.model, train_dataloaders=train_loader,
                         val_dataloaders=val_loader)

    def predict(self, X):
        test_loader = self.prepare_for_test(X)
        predictions = self.trainer.predict(self.model, test_loader)
        ypred = torch.zeros(len(test_loader.dataset),
                            test_loader.dataset[0].y.shape[0])
        idx = 0
        for i, data in enumerate(predictions):            
            bs = int(data.shape[0] / self.n_out_nodes)
            data = data.reshape(bs, self.n_outputs_nodes)
            ypred[idx:idx+bs, :] = data
            idx += bs
            
        return ypred.detach().numpy()

    def score(self, X, y):
        yhat = self.predict(X)
        return mean_absolute_error(y, yhat)
    
    ######################## DATA FORMATING
    def prepare_for_train(self, X, y, Xv=None, yv=None):
        ((X, y), (Xv, yv)) = self.spliter(X, y)
        
        self.convert_indices(X)
        dates, Xe, Xn = self.split_node_edge(X)
        datesv, Xev, Xnv = self.split_node_edge(Xv)

        # ASSUME ALL COUNTRIES HAVE THE SAME NUMBER OF FEATURES
        if (self.nodes_per_country == 24) and (
                (len(self.country_idx[self.country_list[0]]) // 24) == 0):
            raise Exception(
                "Not enough features to reshape into 24 nodes per countries!")
        ns = Xn.shape[0]
        nvs = Xnv.shape[0]
        NUM_WORKERS = len(os.sched_getaffinity(0))
        
        self.last_seen = None
        self.last_connection_matrix = self.get_connection_matrix(dates[0])
        
        start = time.time()       
        # Create the train and val loader
        train_loader = DataLoader(
            [Data(
                x = self.get_nodes(Xn[i, :], dates[i]),
                edge_index = self.get_edges_tensor(dates[i]),
                edge_attr = self.get_edges_attributes(Xe[i, :], dates[i]),
                y = torch.tensor(y[i], dtype=torch.float32))
             for i in range(ns)],
            batch_size=self.batch_size, shuffle=self.shuffle_train,
            num_workers=NUM_WORKERS)

        val_loader = DataLoader(
            [Data(x=self.get_nodes(Xnv[i, :], datesv[i]),
                  edge_index=self.get_edges_tensor(datesv[i]),
                  edge_attr = self.get_edges_attributes(Xe[i, :], dates[i]),
                  y=torch.tensor(yv[i], dtype=torch.float32))
             for i in range(nvs)],
            batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)
        stop = time.time()
        elapsed = stop - start
        print(f"Data Loaders created in {elapsed}s")        
        
        return train_loader, val_loader

    def prepare_for_test(self, X):
        dates, Xe, Xn = self.split_node_edge(X)
        ns = Xn.shape[0]
        
        NUM_WORKERS = len(os.sched_getaffinity(0))
        self.last_seen = None
        self.last_connection_matrix = self.get_connection_matrix(dates[0])
        
        # Create the test_loader (with y = 0)
        test_loader = DataLoader(
            [Data(x=self.get_nodes(Xn[i, :], dates[i]),
                  edge_index=self.get_edges_tensor(dates[i]),
                  edge_attr = self.get_edges_attributes(Xe[i, :], dates[i]),
                  y=torch.tensor(np.zeros(self.N_OUTPUT), dtype=torch.float32))
             for i in range(ns)],
            batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS)

        return test_loader

    def split_node_edge(self, X):
        dates = np.array(
            [datetime.datetime.strptime(d, "%Y-%m-%d").date()
             for d  in X[:, self.psd_idx]])
        Xn = X[:, flatten_dict(self.country_idx)].astype(float)
        Xe = X[:, flatten_dict(self.edges_columns_idx)].astype(float)
        return dates, Xe, Xn

    def convert_indices(self, X):
        """
        Converte node and edge indices to their indices if they are split.
        EX : node indices of FR in X is [1, 2, 3]
        But node indices of FR in Xn is [0, 1, 2]
        """
        converted_node_indices = OrderedDict()
        converted_edge_indices = OrderedDict()

        all_node_indices = flatten_dict(self.country_idx)
        all_edge_indices = flatten_dict(self.edges_columns_idx)        
        
        for country in self.country_list:
            mask = np.zeros(X.shape[1], dtype=bool)
            node_indices = self.country_idx[country]
            mask[node_indices] = True
            converted_node_indices[country] = np.where(mask[all_node_indices])[0]
            
        for edge in self.edges_columns_idx.keys():
            mask = np.zeros(X.shape[1], dtype=bool)            
            edge_indices = self.edges_columns_idx[edge]
            mask[edge_indices] = True
            converted_edge_indices[edge] = np.where(mask[all_edge_indices])[0]
        
        self.converted_node_indices = converted_node_indices
        self.converted_edge_indices = converted_edge_indices
            
    ################## FOR HANDLING NODES    
    def get_nodes(self, X, d):
        filtered = self.filter_countries(X, d)
        reshaped = self.reshape_nodes(filtered, d)        
        return torch.tensor(reshaped, dtype=torch.float32)

    def filter_countries(self, X, d):        
        to_remove = [c for c in self.country_list
                     if c not in self.get_country_list(d)]
        
        if len(to_remove) != 0:
            idx_to_remove = np.concatenate(
                [self.converted_node_indices[tr] for tr in to_remove])
            mask = np.ones(X.shape[0], dtype=bool)
            mask[idx_to_remove] = False
            X = X[mask]
            
        return X

    def countries_have_changed(self, d1, d2):
        if self.is_dynamic:
            raise("Countries might have changed!")
        return False
    
    def get_country_list(self, d):
        country_list = copy.deepcopy(self.country_list)
        if ((self.is_dynamic) and (d < date(2018, 10, 1))
            or (not self.split_DEAT)) and ("AT" in country_list):
            country_list.remove("AT")

        return country_list

    def reshape_nodes(self, X, d):        
        if (self.nodes_per_country == 1):
            # 1 Node per country : The data is in good order :
            # (f1c1, f2c1, f3c1, f1c2, f2c2, f3c2, etc...)
            X = X.reshape(len(self.get_country_list(d)), -1)

        if (self.nodes_per_country == 24):
            # 1 node per hour per country : need to reshape
            nd = X.shape[0]
            nc = len(self.get_country_list(d))
            
            e = int(nd / self.n_nodes)
            nf = int(nd / nc)
            
            X_ = np.zeros((self.n_nodes, e))        
            for j in range(self.n_nodes):
                base = nf * (j // 24)
                inc = j % 24
                X_[j, :] = [X[24 * k + base + inc] for k in range(e)]
            X = X_
            
        return X
    
    ################## FOR HANDLING EDGES    
    def default_connection_matrix(self, d):
        """
        Get the connection matrix of the european network.
        This already filters out connections depending on the graph dynamism,
        the date, and the user choices for treating certain connections.
        """
        matrix = copy.deepcopy(self.matrix)
        if (not self.edge_DEBE) or ((self.is_dynamic) and (d <  date(2020, 11, 5))):
            matrix.loc["BE", "DE"] = 0
            matrix.loc["DE", "BE"] = 0
        
        return matrix

    def get_connection_matrix(self, d):
        """
        Return the connection matrix of the graph for the given day d.
        """
        cl = self.get_country_list(d)
        if (not self.countries_have_changed(self.last_seen, d)) and (self.last_seen is not None):
            return self.last_connection_matrix
        else:
            nc = len(cl)
            matrix = self.default_connection_matrix(d)            
            res = np.zeros((nc, nc))
            for i, c1 in enumerate(cl):
                for j, c2 in enumerate(cl):
                    res[i, j] = matrix.loc[c1, c2]
            self.last_seen = d
            return res
            
    def get_base_edges(self, matrix):
        """
        Return the base edges between countries using the connection matrix.
        """
        edges = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j]:
                    edges.append([i, j])

        return edges

    def get_edges_tensor(self,  d):
        """
        Return the edges for the given date d
        """
        matrix = self.get_connection_matrix(d)
        edges = self.get_base_edges(matrix)

        if self.nodes_per_country == 24:
            # Country edges
            country_edges = []
            for edge in edges:
                edge_from = edge[0]
                edge_to = edge[1]
                for i in range(24):
                    country_edges.append([edge_from * 24 + i, edge_to * 24 + i])

            # Time edges
            time_edges = []
            for i, country in enumerate(self.get_country_list(d)):            
                for h in range(23):
                    time_edges.append([24 * i + h, 24 * i + h + 1])

                time_edges.append([24 * i + 23, 24 * i])
        
            time_edges += self.revert_edges(time_edges)
            edges = country_edges + time_edges
            
        edges_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edges_tensor

    def revert_edges(self, edges):
        reverted = []
        for n1, n2 in edges:
            reverted.append([n2, n1])

        return reverted        

    ###################### HANDLE EDGE ATTRIBUTES
    def get_edges_attributes(self, X, d):
        if not self.is_weighted: return None
        
        filtered = self.filter_edge_countries(X, d)
        reshaped = self.reshape_edges_attributes(filtered, d)
        return torch.tensor(reshaped, dtype=torch.float32)   
    
    def filter_edge_countries(self, X, d):
        """
        Remove edges from countries not present
        """
        country_edges = self.get_connection_matrix(d)
        country_list = self.get_country_list(d)
        idxs = []
        for i, c1 in enumerate(country_list):
            for j, c2 in enumerate(country_list):
                key = f"{c1}_{c2}"
                if country_edges[i, j]:
                    idxs += self.converted_edge_indices[key].tolist()

        return X[idxs]

    def edge_attr_dim(self, X, d):
        if self.nodes_per_country == 1: return 1
        else: return X.shape[1] / self.get_connection_matrix(d).sum()

    def reshape_edges_attributes(self, X, d):
        if (self.nodes_per_country == 1):
            # 1 node per country and average data : 1 attribute per edge
            if (X.shape[0] == self.get_connection_matrix(d).sum()):
                X = X.reshape(-1, 1)
            else:
                # 24 attributes per edges
                X = X.reshape(-1, 24)

        if (self.nodes_per_country == 24):
            # Need to reshape atcs accordingly and put weights in the time edges
            Xcountry = X.reshape(-1, 1)
            Xtime = self.weight_time_edges(d)
            X = np.concatenate((Xcountry, Xtime))

        return X

    def weight_time_edges(self, d):
        weights = []
        # Onwards
        for i, country in enumerate(self.get_country_list(d)):            
            for h in range(23):
                weights.append(self.time_edge_value(d, country, h))
                
            weights.append(self.time_edge_value(d, country, h))
            
        # Backwards
        for i, country in enumerate(self.get_country_list(d)):            
            for h in range(23):
                weights.append(self.time_edge_value(d, country, h))
                
            weights.append(self.time_edge_value(d, country, h))
            
        return np.array(weights).reshape(-1, 1)

    def time_edge_value(self, d, country, h):
        return 0.5

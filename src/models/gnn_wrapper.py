import warnings

from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

from src.models.model_wrapper import *
from src.models.GNN.gnn import GraphNeuralNetwork
from src.models.Splitter import MySplitter

from src.models.sampling.samplers import GNN_space
from src.models.data_scaler import GNNScaler

class GNNWrapper(ModelWrapper):
    def __init__(self, prefix, dataset_name, country="",
                 edge_DEBE=True, split_DEAT=True, is_dynamic=False,
                 replace_ATC="", nodes_per_country=1, spliter=None,
                 predict_two_days=False, n_out=24, known_countries=["CH", "GB"],
                 countries_to_predict="all"):
        ModelWrapper.__init__(self, prefix, dataset_name, country=country,
                              spliter=spliter, predict_two_days=predict_two_days,
                              known_countries=known_countries,
                              countries_to_predict=countries_to_predict,
                              replace_ATC=replace_ATC)
        if spliter is None: spliter = MySplitter(0.25)
        self.spliter = spliter
        self.external_spliter = None        
        self.validation_mode = "internal"
        self.nodes_per_country = nodes_per_country
        self.n_out_per_nodes = int(n_out / nodes_per_country)
        self.n_out_per_edges = int(n_out / nodes_per_country)        
        self.n_out = n_out
        
        self.is_dynamic = is_dynamic
        self.edge_DEBE = edge_DEBE
        self.split_DEAT = split_DEAT
        
        self.known_countries = known_countries
        self.countries_to_predict = countries_to_predict
        
        # Load ATC columns even if not weighted
        self.edges_columns = self.load_atc_columns()    

        # Remove known countries and undesired labels
        self.update_labels()
        self.update_columns()
        
        #For loading connections. Has to be defined here becasue,its not possible
        # to access os.environ in multi-threading context!!!
        self.connection_path = os.path.join(
            os.environ["OPALE"],"data","datasets","CONNECTIONS.csv")        

    def make(self, ptemp):
        scaler, transformer, ptemp_ = self.prepare_for_make(ptemp, GNN="keep")
        model = GraphNeuralNetwork("test", ptemp)        
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)
        return regr
    
    def predict_val(self, regr, X, y=None, oob=False):
        if oob: print("Can't access the oob prediction!")
        return self.predict(regr, X)
        
    def eval_val(self, regr, X, y, oob=False):
        """
        Use the out of sample validation loss to provide an estimate of the 
        generalization error. 
        """
        if not oob:
            yhat = self.predict_val(regr, X, oob=oob)
            return mean_absolute_error(y, yhat)                
        else:
            scaled_loss = regr.regressor_.steps[1][1].callbacks[0].val_losses[-1]
            return scaled_loss    
        
    def params(self):
        return {
            # Graph Structure params
            "country_idx" : self.country_idx,
            "is_dynamic" : self.is_dynamic,
            "psd_idx" : self.psd_idx,
            "edges_columns_idx" : self.edges_columns_idx,
            "edge_DEBE" : self.edge_DEBE,
            "split_DEAT" : self.split_DEAT,
            "known_countries" : self.known_countries,
            "countries_to_predict" : self.countries_to_predict,            
            "nodes_per_country" : self.nodes_per_country,            
            "n_out_per_nodes" : self.n_out_per_nodes,
            "n_out_per_edges" : self.n_out_per_edges,            
            
            # Networw architecture
            "N_OUTPUT" : len(self.label),
            "channels" : (50, ),
            "edge_channels" : 24,
            "regr" : [],
            "edge_regr" : [],
            "all_channels" : None,
            "activation" : "relu",
            "dropout" : 0.0,
            "batch_norm" : True,            
            "pooling" : "mean",
            "activation" : "ReLU",
            "layer_type" : "GCNConv",
            "heads" : 5,            

            # Training params
            "spliter" : self.spliter,            
            "n_epochs" : 100000,
            "batch_size" : 30,
            "early_stopping" : "sliding_average",
            "early_stopping_alpha" : 20,
            "early_stopping_patience" : 20,    
            "adapt_lr" : False,
            "shuffle_train" : True,            
            
            # Optimizer params
            "optimizer" : "Adam",
            "learning_rate" : 0.001,            
            "criterion" : "HuberLoss",

            # Pipeline Params
            "scaler" : "BCM",
            "transformer" : "Standard",

            # Other
            "connection_path" : self.connection_path,
        }  

    def map_dict(self):
        orig = ModelWrapper.map_dict(self)
        orig.update({"channels" : (mu.neurons_per_layer_to_string,
                                   mu.neurons_per_layer_from_string),
                     "regr" : (mu.neurons_per_layer_to_string,
                               mu.neurons_per_layer_from_string)})
        return orig
    
    def get_val_losses(self, regr):
        return regr.regressor_.steps[1][1].callbacks[0].val_losses

    def get_search_space(self, country, version=None,  n=None, fast=False,
                         stop_after=-1):
        return GNN_space(n, country, fast=fast, stop_after=stop_after)

    def string(self):
        return "GNN"    

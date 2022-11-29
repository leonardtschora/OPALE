from tensorflow.keras import initializers
import tensorflow.keras.backend as K

from src.models.model_wrapper import *
from src.models.Splitter import MySplitter
import src.models.parallel_scikit as ps

from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor

from src.models.NN.nn import DNN, CNN
from src.models.sampling.samplers import DNN_space, CNN_space

class NeuralNetWrapper(ModelWrapper):
    def __init__(self, prefix, dataset_name, country="", spliter=None,
                 predict_two_days=False, flow_estimation="",
                 known_countries=["CH", "GB"], countries_to_predict="all"):
        ModelWrapper.__init__(self, prefix, dataset_name, country=country,
                              spliter=spliter, predict_two_days=predict_two_days,
                              known_countries=known_countries,
                              flow_estimation=flow_estimation,
                              countries_to_predict=countries_to_predict)
        if spliter is None: spliter = MySplitter(0.25)        
        self.spliter = spliter
        self.external_spliter = None        
        self.validation_mode = "internal"    

    def params(self):
        return {
            "N_OUTPUT" : len(self.label),
    
            "batch_norm" : False,
            "batch_norm_epsilon" : 10e-7,

            "neurons_per_layer" : (240, 160, ),
            "activations" : ("relu", ),            
            "dropout_rate" : 0.25,
    
            "learning_rate" : 0.001,
            "optimizer" : "Adam",
            "loss_function" : "LogCosh",
            "metrics" : ["mean_absolute_error"],
            "use_bias" : True,
            
            "default_kernel_initializer" : initializers.he_uniform(),
            "out_layer_kernel_initializer" : initializers.lecun_uniform(),
            "default_bias_initializer" : initializers.Zeros(),
            "out_layer_bias_initializer" : initializers.Zeros(),
            "default_activity_regularizer" : None,    
            
            "adapt_lr" : False,        
            "n_epochs" : 100000,
            "batch_size" : 300,
            "early_stopping" : "sliding_average",
            "early_stopping_alpha" : 25,
            "stop_after" : -1,
            "stop_threshold": -1,
            "shuffle_train" : True,
            "scaler" : "BCM",
            "transformer" : "Standard",

            "spliter" : self.spliter,
        }

    def map_dict(self):
        orig = ModelWrapper.map_dict(self)
        orig.update({
            "default_activity_regularizer" :
            (mu.activity_regularizer_to_string,
             mu.activity_regularizer_from_string),
            "neurons_per_layer" :
            (mu.neurons_per_layer_to_string,
             mu.neurons_per_layer_from_string)
        })
        return orig
    
    def save(self, regr):
        print("Can't save a Pipeline containing keras models because they are not pickable")
        print("Saving the configuration instead. Loading will be expensive because it will retrain")
        
        all_params = regr.regressor.steps[1][1].get_params()
        joblib.dump(all_params, self.model_path())

    def load(self):        
        all_params = joblib.load(self.model_path())
        regr = self.make(all_params["model_"])

        print("Fitting with the test dataset")
        X, y = self.load_train_dataset()
        ps.set_all_seeds(all_params["model_"]["seeds"])
        regr.fit(X, y)
        return regr
    
    def predict_test(self, regr, X):
        K.clear_session()
        return self.predict_test_(regr, X)

class DNNWrapper(NeuralNetWrapper):
    def __init__(self, prefix, dataset_name, country="",
                 spliter=None, predict_two_days=False, flow_estimation="",
                 known_countries=["CH", "GB"], countries_to_predict="all"):
        NeuralNetWrapper.__init__(self, prefix, dataset_name, country=country,
                                  spliter=spliter, flow_estimation=flow_estimation,
                                  predict_two_days=predict_two_days,
                                  known_countries=known_countries,
                                  countries_to_predict=countries_to_predict)
        
    def params(self):
        orig = NeuralNetWrapper.params(self)
        orig.update()
        return orig

    def map_dict(self):
        orig = NeuralNetWrapper.map_dict(self)
        orig.update({})
        return orig       

    def make(self, ptemp):
        scaler, transformer, ptemp_ = self.prepare_for_make(ptemp)
            
        model = DNN("test", ptemp_)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)

        return regr
    
    def predict_val(self, regr, X, oob=False):
        if oob: print("Can't access the oob prediction!")
        else: return self.predict(regr, X)
        
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
        
    def string(self):
        return "DNN"  
    
    def get_search_space(self, country, version=None,  n=None, fast=False,
                         stop_after=-1):
        return DNN_space(n, country, fast=fast, stop_after=stop_after)    

class CNNWrapper(NeuralNetWrapper):
    def __init__(self, prefix, dataset_name, country="", flow_estimation="",
                 spliter=None, predict_two_days=False, W=None, H=24,
                 known_countries=["CH", "GB"], countries_to_predict="all"):
        NeuralNetWrapper.__init__(self, prefix, dataset_name, country=country,
                                  spliter=spliter, flow_estimation=flow_estimation,
                                  predict_two_days=predict_two_days,
                                  known_countries=known_countries,
                                  countries_to_predict=countries_to_predict)
        self.unreshapable_cols = self.load_unreshapable_cols()
        self.H = 24        
        self.W = int((len(self.columns) - len(self.unreshapable_cols)) / 24)        

    def load_unreshapable_cols(self):
        return mu.load_columns(self.dataset_name, order_str="unreshapable")
        
    def params(self):
        orig = NeuralNetWrapper.params(self)
        orig.update(
            {"conv_activation" : "relu",
             "filter_size" : ((6, ), (16, ), (24, )),
             "dilation_rate" : (((1, 1), ), ((1, 1), ), ((1, 1), )),             
             "kernel_size" : (((5, 5), ), ((5, 5), ), ((5, 5), )), 
             "pool_size" : ((2, 2), (2, 2), (2, 2)),
             "strides" : ((2, 2), (2, 2), (2, 2)),
             
             "neurons_per_layer" : (),
        })
        return orig

    def best_params(self, df, for_recalibration=False, acc=False,
                    filters={}, inverted_filters={}, recompute=True):        
        df = self.filter_results(copy.deepcopy(df), filters, inverted_filters)

        # Use this specific config for recalibration : other configurations are too
        # computiationally expensive...
        best_row = df.seeds.argmax()
        best_params = df.loc[best_row].to_dict()        
        print(f"BEST MAE = {round(best_params['maes'], ndigits=2)}")            
        
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
            

    def map_dict(self):
        orig = NeuralNetWrapper.map_dict(self)
        orig.update({"structure" :
                     {
                         "filter_size" : (mu.filter_size_to_string,
                                          mu.filter_size_from_string),
                         "dilation_rate" : (mu.dilation_rate_to_string,
                                            mu.dilation_rate_from_string),
                         "kernel_size" : (mu.dilation_rate_to_string,
                                          mu.dilation_rate_from_string),
                         "pool_size" : (mu.dilation_rate_to_string,
                                        mu.dilation_rate_from_string),
                         "strides" : (mu.neurons_per_layer_to_string,
                                      mu.neurons_per_layer_from_string),
                     }                    
        })
        return orig

    def make(self, ptemp):
        scaler, transformer, ptemp_ = self.prepare_for_make(ptemp)
        
        model = CNN("", ptemp_, self.W, self.H)
        pipe = make_pipeline(scaler, model)
        regr = TransformedTargetRegressor(pipe, transformer=transformer)

        return regr             

    def load_dataset(self, path):
         if self.countries_to_predict_ == "not_graph":
             # Drops every non-reshape-able columns
             dataset = pandas.read_csv(path)
             labels = dataset[np.array(self.label)]
             dataset.drop(
                 columns=self.label+self.unreshapable_cols+["period_start_date"],
                 inplace=True)
             names = [c for c in self.columns if c not in self.unreshapable_cols]
             
             # Sort the names
             dataset = dataset[names]
             self.past_label_idx = mu.past_label_columns(
                 dataset, self.past_label)        
             X = dataset.values 
             y = labels.values        
             return X, y
         else:
             # Graph datasets are fully reshapable /24 by nature!
             return self.load_graph_dataset(path)
    
    def predict_val(self, regr, X, oob=False):
        if oob:
            print("Can't access the oob prediction!")
        else:
            return self.predict(regr, X)
    
    def eval_val(self, regr, X, y, oob=False):
        """
        Use the out of sample validation loss to provide an estimate of the 
        generalization error. 
        """
        if not oob:
            return self.eval(regr, X, y)
        else:
            scaled_loss = regr.regressor_.steps[1][1].callbacks[0].val_losses[-1]
            return scaled_loss

    def get_search_space(self, country=None, version=None, n=None,
                         fast=False, stop_after=-1):
        return CNN_space(n, self.W, self.H, country, fast=fast,
                         stop_after=stop_after)

    def string(self):
        return "CNN"

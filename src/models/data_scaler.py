import numpy as np, copy, warnings
from statsmodels.robust import mad
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from src.models.model_utils import flatten_dict

class DataScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then appply the arcsinh.
    """
    def __init__(self, scaling, spliter=None, GNN="",
                 psd_idx=None, country_idx=None, edges_columns_idx=None):
        self.scaling = scaling
        self.spliter = spliter
        self.GNN = GNN
        self.psd_idx = psd_idx
        self.country_idx = country_idx
        self.edges_columns_idx = edges_columns_idx

        if GNN != "":
            self.scaler = GNNScaler(scaling, psd_idx, country_idx,
                                    edges_columns_idx, GNN)
        else:
            if self.scaling == "BCM":
                self.scaler = BCMScaler()
            if self.scaling == "Standard":
                self.scaler = StandardScaler()
            if self.scaling == "Median":
                self.scaler = MedianScaler()
            if self.scaling == "SinMedian":
                self.scaler = SinMedianScaler()
            if self.scaling == "InvMinMax":
                self.scaler = InvMinMaxScaler()
            if self.scaling == "MinMax":
                self.scaler = MinMaxScaler()
            if self.scaling not in ("BCM", "Standard", "", "Median",
                                    "SinMedian", "InvMinMax", "MinMax"):
                raise ValueError('Scaling parameter must be one of "BCM", "Standard", "", Median, SinMedian, InvMinMax, MinMax!')
            
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        
        if self.spliter is not None:
            (X, _) = self.spliter(X)
            
        if not self.scaling == "":
            self.scaler.fit(X)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        if self.scaling == "": return X
        else: return self.scaler.transform(X)

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        if self.scaling == "": return X
        else: return self.scaler.inverse_transform(X)


class BCMScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]
        self.scaler.fit(X)        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.transform(X)
        transformed_data = np.arcsinh(transformed_data)
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        transformed_data = np.sinh(np.float128(X))        
        transformed_data = np.float32(
            self.scaler.inverse_transform(transformed_data))

        # Need to post-treat infinity in case....
        inf_idx = np.where(transformed_data == np.inf)[0]
        if len(inf_idx) > 0:
            warnings.warn("Infinity in the output!!!!!")        
        transformed_data[inf_idx] = np.sinh(80)

        return transformed_data


class MedianScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Median Scaler
    """
    def __init__(self, epsilon=10e-5):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        self.median = np.median(X, axis=0)
        self.mad = mad(X, axis=0)
        self.mad = np.clip(self.mad, a_min=self.epsilon, a_max=None)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = (X - self.median) / self.mad
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        transformed_data = (X * self.mad) + self.median        
        return transformed_data


class SinMedianScaler(TransformerMixin, BaseEstimator):
    """
    Standardize the data using a Standard Scaler, then apply the arcsinh.
    """
    def __init__(self):
        self.scaler = MedianScaler()

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]
        self.scaler.fit(X)        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.transform(X)
        transformed_data = np.arcsinh(transformed_data)
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        transformed_data = np.sinh(np.float128(X))        
        transformed_data = np.float32(
            self.scaler.inverse_transform(transformed_data))

        # Need to post-treat infinity in case....
        inf_idx = np.where(transformed_data == np.inf)[0]
        if len(inf_idx) > 0:
            warnings.warn("Infinity in the output!!!!!")
        transformed_data[inf_idx] = np.sinh(80)
        
        return transformed_data

class InvMinMaxScaler(TransformerMixin, BaseEstimator):
    """
    Highest value is mapped to 0 and lowest to 1
    """
    def __init__(self, epsilon=10e-5):
        self.scaler = MinMaxScaler()
        self.epsilon = epsilon

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        transformed_data = 1 / np.clip(X, a_min=self.epsilon, a_max=None)
        self.scaler.fit(transformed_data)
        
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = 1 / np.clip(X, a_min=self.epsilon, a_max=None)
        transformed_data = self.scaler.transform(transformed_data)
        
        return transformed_data

    def inverse_transform(self, X, y=None):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = self.scaler.inverse_transform(transformed_data)
        transformed_data = 1 / np.clip(transformed_data, a_min=self.epsilon, a_max=None)

        return transformed_data

class ZeroMinMaxScaler(TransformerMixin, BaseEstimator):
    """
    A Min Max Scaler where 0s are excluded from the min computation.
    """
    def __init__(self, min_value=0.1, default_value=0.5, epsilon=10e-5):
        self.epsilon = epsilon
        self.min_value = min_value
        self.default_value = default_value

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]
        self.mins_ = np.zeros(self.n_features_)
        self.maxs_ = np.zeros(self.n_features_)
        
        for i in range(self.n_features_):
            ind_pos = np.where(X[:, i] > 0)[0]
            data = X[ind_pos, i]

            if len(data) > 0:
                self.mins_[i] = data.min()
                self.maxs_[i] = data.max()
            
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')
        X = check_array(X, accept_sparse=True)        
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        
        transformed_data = copy.deepcopy(X).astype(object)
        for i in range(self.n_features_):
            ind_pos = np.where(X[:, i] > 0)[0]
            data = X[ind_pos, i]
            # If we found non-zero data
            if len(data) > 0:
                if self.maxs_[i] > 0:
                    transformed_data[ind_pos, i] = np.clip(
                        (data - self.mins_[i]) / (
                            self.maxs_[i] - self.mins_[i] + self.epsilon),
                        self.min_value, 1)                    
                else:
                    # If the recorded max was 0 but non zero data arrives
                    transformed_data[ind_pos, i] =  self.default_value * np.ones(
                        len(ind_pos))
                    
        return transformed_data

    def inverse_transform(self, X, y=None):
        raise ValueError("This is for ATC only so no inverse transform is required")
    
class GNNScaler(TransformerMixin, BaseEstimator):
    """
    Standardizer for the GNN data.
    This scaler first leave the date column untouched.
    Then, it separates the Nodes data from the edges data.
    The nodes data is normalized using the node_scaler.
    The edge data is normalized using the edge_scaler which is 
    """
    def __init__(self, nodes_scaler, psd_idx, node_columns_idx,
                 edges_columns_idx, GNN):
        self.nodes_scaler = DataScaler(nodes_scaler, spliter=None)
        self.edges_scaler = ZeroMinMaxScaler()
        self.psd_idx = psd_idx
        self.node_columns_idx = flatten_dict(node_columns_idx)
        self.edges_columns_idx = flatten_dict(edges_columns_idx)
        self.GNN = GNN
        
    def split_node_edge(self, X):        
        dates = X[:, self.psd_idx]
        Xe = X[:, self.edges_columns_idx]
        Xn = X[:, self.node_columns_idx]    
        return dates, Xe, Xn

    def merge_node_edge(self, dates, Xe, Xn):
        X = np.empty((len(dates), Xe.shape[1] + Xn.shape[1] + 1), dtype='object')
        X[:, self.psd_idx] = dates
        X[:, self.node_columns_idx] = Xn
        X[:, self.edges_columns_idx] = Xe
        return X
    
    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        dates, Xe, Xn = self.split_node_edge(X)

        self.zero_edges = np.where(Xe.mean(axis=0) == 0)[0]
        self.zero_nodes = np.where(Xn.mean(axis=0) == 0)[0]
        
        self.nodes_scaler.fit(Xn)
        if Xe.shape[1] != 0: self.edges_scaler.fit(Xe)
            
        self.is_fitted_ = True
        return self        
    
    def transform(self, X):
        check_is_fitted(self, 'n_features_')      
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')

        dates, Xe, Xn = self.split_node_edge(X)        
        Xn = self.nodes_scaler.transform(Xn)
        
        if Xe.shape[1] != 0:        
            Xe = self.edges_scaler.transform(Xe)

        # Refill with 0s
        Xe[:, self.zero_edges] = 0
        Xn[:, self.zero_nodes] = 0        
            
        # Merge date, transformed_nodes and transformed_edges
        X = self.merge_node_edge(dates, Xe, Xn)
        if self.GNN == "drop_date":
            X = copy.deepcopy(X)[:, np.sort(np.concatenate(
                (self.edges_columns_idx, self.node_columns_idx)))]
            X = X.astype(np.float64)
            
        return X

    def inverse_transform(self, X, y=None):
        raise("Can't use a GNNScaler to unscale!")
    

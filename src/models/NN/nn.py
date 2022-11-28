import copy, tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

import tensorflow_addons as tfa
from tensorflow.keras import optimizers, metrics, losses, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.layers import BatchNormalization

from src.models.NN.callbacks import *


class NeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(self, name, model_):
        self.name = name
        self.model_ = model_
        self.N_OUTPUT = model_["N_OUTPUT"] 

        # Layers parameters (for the last layers in a CNN)
        self.neurons_per_layer = model_["neurons_per_layer"]
        self.activations = model_["activations"]
        
        self.batch_norm = model_["batch_norm"]
        self.batch_norm_epsilon = model_["batch_norm_epsilon"]
        self.dropout_rate = model_["dropout_rate"]                
        self.use_bias = model_["use_bias"]

        # Initializers
        self.default_kernel_initializer = copy.deepcopy(
            model_["default_kernel_initializer"])
        self.out_layer_kernel_initializer = copy.deepcopy(
            model_["out_layer_kernel_initializer"])
        self.default_bias_initializer = copy.deepcopy(
            model_["default_bias_initializer"])
        self.out_layer_bias_initializer = copy.deepcopy(
            model_["out_layer_bias_initializer"])

        # Regularizers
        self.default_activity_regularizer = copy.deepcopy(
            model_["default_activity_regularizer"])
        
        # Gradient descent parameters
        self.learning_rate = model_["learning_rate"]
        self.optimizer = model_["optimizer"]
        self.loss = model_["loss_function"]
        self.metrics = model_["metrics"]
        self.adapt_lr = model_["adapt_lr"]

        # Fit parameters
        self.n_epochs = model_["n_epochs"]
        self.batch_size = model_["batch_size"]
        self.early_stopping = model_["early_stopping"]
        self.early_stopping_alpha = model_["early_stopping_alpha"]
        self.shuffle_train = model_["shuffle_train"]

        self.stop_after = model_["stop_after"]
        self.stop_threshold = model_["stop_threshold"]

        # Spliter
        self.spliter = model_["spliter"]

    def set_params(self, **parameters):
        for parameter, value in parameters.items():            
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y, verbose=0):
        self.update_params()

        # Resplit the data
        ((X, y), (Xv, yv)) = self.spliter(X, y)
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size,
                       callbacks=self.callbacks, validation_data=(Xv, yv),
                       shuffle=self.shuffle_train, verbose=verbose)
        
        return self
    
    def predict(self, X):
        return self.model.predict_step(X)
    
    def score(self, X, y):
        yhat = self.predict(X)
        return mean_absolute_error(y, yhat)        

    def update_params(self, input_shape=None):
        """
        Update the mlp's networks and the model's callback when a parameter changes
        """
        # Set callbacks
        self.callbacks = []
        self.early_stopping_callbacks()
        self.adapt_lr_callbacks()

        # Instantiate the model
        self.model = self.create_network(input_shape=input_shape)
        
    def create_network(self, input_shape=None):
        raise("Not Implemented by default!")

    def build_layers(self, model=None, input_shape=None):
        # Add layers on top of each other. If the batch norm has to be used,
        # then batch normalization layers will be added between each layers.
        try:
            len(self.neurons_per_layer)
            try:
                self.neurons_per_layer[0]
            except:
                pass
        except:
            self.neurons_per_layer = (self.neurons_per_layer, )

        if model is None: model = AddList()
        for i, neuron in enumerate(self.neurons_per_layer):
            try:
                activation = self.activations[i]
            except:
                activation = self.activations[0]

            try:
                kernel_initializer = self.default_kernel_initializer[i]
            except:
                kernel_initializer = self.default_kernel_initializer
                
            try:
                bias_initializer = self.default_bias_initializer[i]            
            except:
                bias_initializer = self.default_bias_initializer

            try:
                activity_regularizer = self.default_activity_regularizer[i]
            except:
                activity_regularizer = self.default_activity_regularizer
                
            layer = Dense(
                neuron, activation=activation,
                kernel_initializer=copy.deepcopy(kernel_initializer),
                bias_initializer=copy.deepcopy(bias_initializer),
                activity_regularizer=copy.deepcopy(activity_regularizer),
                use_bias=self.use_bias)

            if self.dropout_rate > 0.0:
                model.add(Dropout(self.dropout_rate))
            if self.batch_norm:         
                model.add(BatchNormalization(
                    epsilon=self.batch_norm_epsilon))
                
            model.add(layer)
            
        # Add the output layers
        if self.dropout_rate > 0.0:
            model.add(Dropout(self.dropout_rate))        
        if self.batch_norm:
            model.add(BatchNormalization(epsilon=self.batch_norm_epsilon))
            
        output_layer = Dense(self.N_OUTPUT, activation='linear',
                                    kernel_initializer=copy.deepcopy(
                                        self.out_layer_kernel_initializer),
                                    bias_initializer=copy.deepcopy(
                                        self.out_layer_bias_initializer),
                                    use_bias=self.use_bias)
        model.add(output_layer)
        return model

    def tboard_callbacks(self):
        self.callbacks.append(tboard())       
        
    def adapt_lr_callbacks(self):
        if self.adapt_lr:
            self.callbacks.append(
                ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=int(self.n_epochs/100),
                                  verbose=0, mode='min', min_delta=0.0,
                                  cooldown=0, min_lr=0.0001))
                
    def early_stopping_callbacks(self):
        if self.early_stopping:
            alpha = self.early_stopping_alpha
            if self.early_stopping == "best_epoch":
                self.callbacks.append(EarlyStoppingBestEpoch(
                    monitor='val_loss', verbose=0))
                
            if self.early_stopping == "decrease_val_loss":
                self.callbacks.append(EarlyStoppingDecreasingValLoss(
                    monitor='val_loss', patience=int(self.n_epochs/10),
                    verbose=0, restore_best_weights=True))
                
            if (self.early_stopping == "sliding_average"
                or self.early_stopping == "both"):
                self.callbacks.append(EarlyStoppingSlidingAverage(
                    monitor = 'val_loss', patience=alpha,
                    verbose=0, alpha=alpha, restore_best_weights=True))
                
            if (self.early_stopping == "prechelt"
                or self.early_stopping == "both"):
                self.callbacks.append(PrecheltEarlyStopping(
                    monitor = 'loss', val_monitor = 'val_loss',
                    baseline = 2.5, verbose=0, alpha=alpha))

            if self.stop_after > 0:
                if self.stop_threshold > 0:
                    self.callbacks.append(TimeStoppingAndThreshold(
                        seconds=self.stop_after, verbose=0,
                        threshold=self.stop_threshold,
                        monitor="val_loss"))
                else:
                    self.callbacks.append(tfa.callbacks.TimeStopping(
                        seconds=self.stop_after, verbose=0))
                    
class DNN(NeuralNetwork):
    def __init__(self, name, model_):
        NeuralNetwork.__init__(self, name, model_)

    def create_network(self, input_shape=None):
        model = tf.keras.Sequential()
        self.build_layers(model, input_shape=input_shape)
        model.compile(
            optimizer=getattr(optimizers, self.optimizer)(),
            metrics=[getattr(metrics, metric) for metric in self.metrics],
            loss=getattr(losses, self.loss)(), run_eagerly=True)
        return model

class CNN(NeuralNetwork):
    def __init__(self, name, model_, W, H):
        NeuralNetwork.__init__(self, name, model_)
        
        self.conv_activation = model_["conv_activation"]
        self.filter_size = model_["filter_size"]
        self.dilation_rate = model_["dilation_rate"]
        self.kernel_size = model_["kernel_size"]
        self.pool_size = model_["pool_size"]
        self.strides = model_["strides"]

        self.W = W
        self.H = H
        
    def create_network(self, input_shape=None):
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([Input(shape=input_shape)])
        conv_activation = self.conv_activation
        for i in range(len(self.filter_size)):
            for j in range(len(self.filter_size[i])):
                filter_size = self.filter_size[i][j]
                dilation_rate = self.dilation_rate[i][j]
                kernel_size = self.kernel_size[i][j]
                model.add(Conv2D(filter_size, kernel_size=kernel_size,
                                 padding="same",
                                 dilation_rate=dilation_rate,
                                 activation=conv_activation))
                
            pool_size = self.pool_size[i]
            strides = self.strides[i]
            if pool_size is not None:
                model.add(MaxPooling2D(pool_size=pool_size, strides=strides))
        
        model.add(Flatten())
        self.build_layers(model)
        model.compile(
            optimizer=getattr(optimizers, self.optimizer)(),
            metrics=[getattr(metrics, metric) for metric in self.metrics],
            loss=getattr(losses, self.loss)(), run_eagerly=True)

        flatten = model.get_layer("flatten")
        nout = np.array(flatten.input.shape[1:]).prod()
        if nout > 1000:
            s=f"Configuration is too big: {np.array(flatten.input.shape[1:])} * {792} = {nout * 792} outputs!"
            print(s)
            raise Exception(s)
            
        return model

    def prepare_for_train(self, X, y, Xv=None, yv=None):
        # Reshape the input data
        ((X, y), (Xv, yv)) = self.spliter(X, y)
        
        X = X.reshape(-1, self.W, self.H)
        X = np.expand_dims(X, -1)

        Xv = Xv.reshape(-1, self.W, self.H)
        Xv = np.expand_dims(Xv, -1)        

        return (X, y, Xv, yv)

    def prepare_for_test(self, X):
        X = X.reshape(-1, self.W, self.H)
        X = np.expand_dims(X, -1)
        return X    
    
    def fit(self, X, y, verbose=0, Xv=None, yv=None):
        (X, y, Xv, yv) = self.prepare_for_train(X, y)
        self.update_params(input_shape=X.shape[1:])        
        self.model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size,
                       callbacks=self.callbacks, validation_data=(Xv, yv),
                       shuffle=self.shuffle_train, verbose=verbose)
        return self
    
    def predict(self, X):
        X = self.prepare_for_test(X)
        return tf.reshape(self.model.predict_step(X), (-1, self.N_OUTPUT))


class AddList(object):
    def __init__(self):
        self.list_ = []

    def add(self, e):
        self.list_.append(e)

    def chain_list(self, previous_layer):
        for layer in self.list_:
            previous_layer = layer(previous_layer)

        return previous_layer    

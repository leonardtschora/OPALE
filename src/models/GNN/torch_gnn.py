import pandas, torch, numpy as np
import pytorch_lightning as pl

import torch.nn.functional as F
from torch.nn import Dropout, Linear, Sequential
from torch import optim, nn, flatten
from torch_geometric import nn as tgnn
from torch_geometric.nn import BatchNorm

class TorchGNN(pl.LightningModule):
    def __init__(self, batch_size, channels, activation, input_shape,
                 output_shape, criterion, optimizer, learning_rate, layer_type,
                 edge_shape, n_nodes, n_edges,
                 n_out_per_nodes, n_out_nodes, test_mask,                 
                 heads=1, regr=[], batch_norm=True, lam=0.5,
                 dropout=0.5, pooling="mean", edge_channels=None):
        pl.LightningModule.__init__(self)
        torch.manual_seed(12345)
        self.batch_size = batch_size
        self.edge_shape = edge_shape
        self.input_shape = input_shape
        self.n_nodes = n_nodes        
        self.output_shape = output_shape
        
        self.n_edges = n_edges
        self.channels = channels
        self.edge_channels = edge_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.regr = list(regr)
        self.pooling = pooling
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.activation = activation
        self.pooling = pooling
        self.layer_type = layer_type
        self.heads = heads
        self.lam = lam

        self.criterion_ = getattr(nn, self.criterion)()
        self.layers = []
        self.n_out_nodes = n_out_nodes
        self.n_out_per_nodes = n_out_per_nodes
        
        # Initialize test masks
        self.test_mask = test_mask
        self.regular_test_mask = self.compute_test_mask(self.batch_size)

        # Construct layers 
        channels = [input_shape] + list(channels)
        edge_channels = [edge_shape] + list(edge_channels)        
        for i, (c1, c2) in enumerate(zip(channels[:-1], channels[1:])):
            (e1, e2) = (edge_channels[i], edge_channels[i+1])            
            self.add_conv_layer(i, c1, c2, e1=e1, e2=e2)
        
            # Add BNorm, Dropout, Activation if specified
            if i != len(channels):            
                self.add_bn_layer(i, c2)
                self.add_dropout(i)
                self.add_activation(i)

        # Case where no regressor : use a last conv layer as final layer
        # with only 1 head if gatv2 conv
        regr = c2
        if len(self.regr) == 0:
            self.add_conv_layer(i+1, regr, self.n_out_per_nodes, e1=e2, e2=1)
        else:            
            self.regr_layers = np.zeros(
                (self.n_out_nodes, len(self.regr) + 1),dtype=object)
            for z in range(self.n_out_nodes):
                for (j, (c1, c2)) in enumerate(zip(
                        [regr] + self.regr, self.regr + [self.n_out_per_nodes])):
                    self.regr_layers[z, j] = []
                    self.add_mlp_layer(j, c1, c2, z)
                    
                    #Add BN, Dropout and activation if this is not the last layer
                    if j != len(self.regr):
                        self.add_regr_bn_layer(j, c2, z)
                        self.add_regr_dropout(j, z)
                        self.add_regr_activation(j, z)

    def build_layer(self, c1, c2, heads=None, e1=None, e2=None):
        if heads is None:
            heads = self.heads
        
        layer_type = getattr(tgnn, self.layer_type)
        if self.layer_type == "GCNConv":
            return layer_type(c1, c2)
        elif self.layer_type == "GINConv":
            nn_config = Sequential(Linear(c1, c2))            
            return layer_type(nn_config, train_eps=True)
        elif self.layer_type == "GeneralConv":
            return layer_type(c1, c2, in_edge_channels=self.edge_shape,
                              directed_msg=False)
        elif self.layer_type == "NNConv":
            nn_config = Sequential(Linear(self.edge_shape, c1 * c2))
            return layer_type(c1, c2, nn_config)
        elif self.layer_type == "GATv2Conv":
            return layer_type(c1, c2, heads=heads,
                              edge_dim=self.edge_shape)
        else:
            raise Exception(
                f"Layer type {self.layer_type} is not supported (YET)")
        
    def forward_(self, x, edge_index, edge_attr, batch):
        # Get the edge batch indices
        row, col = edge_index
        edge_batch = batch[row]
        
        # Call the created layers 1 by 1
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            # Call the CONV layers
            if "CONV" in layer_name:
                # Update both node  embeddings
                x = layer(x, edge_index, edge_attr)
            elif (("POOLING_NODE" in layer_name) or (layer_name == "POOLING")):
                # Call the NODE Pooling layer
                x = layer(x, batch)
            elif ("POOLING_EDGE" in layer_name):
                # Call the EDGE Pooling layer
                edge_attr = layer(edge_attr, edge_batch)
            elif "EDGE" in layer_name:
                # Case of NE Graph OR NELayer:apply the layer to the edge embeddings
                edge_attr = layer(edge_attr)
            else:
                # Otherwise, apply to the nodes embeddings            
                x = layer(x)
        
        return (x, edge_attr)

    def forward(self, data):
        (nodes, edges) = self.forward_(
            data.x, data.edge_index, data.edge_attr, data.batch)
        bs = int(data.x.shape[0] / self.n_nodes)
        test_mask = self.get_test_mask(bs)
        
        nodes = nodes[test_mask, :]        
        nodes = self.forward_regr(nodes)
        return nodes    

    def forward_regr(self, x):
        # Apply the regressor for each node if specified
        if len(self.regr) == 0:
            return x
        else:
            bs = int(x.shape[0] / self.n_out_nodes)

            # Intialize the out vector of siwe bs * output_shape
            out = torch.tensor(np.zeros(
                (bs * self.n_out_nodes, self.n_out_per_nodes),
                dtype=np.float32))

            # Apply regressor to each node separatly
            for z in range(self.regr_layers.shape[0]):

                # Compute the indices of a given node
                indices = [ib * self.n_out_nodes + z
                           for ib in range(bs)]
                input_data = x[indices, :]

                # Apply each regressor layers
                for i in range(self.regr_layers.shape[1]):
                    layers = self.regr_layers[z, i]
                    for layer_name in layers:
                        layer = getattr(self, layer_name)
                        input_data = layer(input_data)

                # Insert the output in the predictions
                out[indices, :] = input_data
        return out    

    def _step(self, data, batch_idx):
        out = self.forward(data)
        loss = self.criterion_(out.reshape_as(data.y), data.y)
        bs = int(data.x.shape[0] / self.n_nodes)
        return loss, bs
    
    def training_step(self, data, batch_idx):
        loss, bs = self._step(data, batch_idx)
        self.log("epoch_loss", loss, batch_size=bs, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, data, batch_idx):
        loss, bs = self._step(data, batch_idx)  
        self.log("val_loss", loss, batch_size=bs, logger=True, on_epoch=True)    

    def configure_optimizers(self):
        return getattr(optim, self.optimizer)(
            self.parameters(), lr=self.learning_rate)

    def add_bn_layer(self, i, s):
        if self.batch_norm:
            bn_name = "BN_" + str(i)
                
            if self.layer_type == "GATv2Conv":
                s *= self.heads
            setattr(self, bn_name, BatchNorm(s))
            self.layers += [bn_name]
            
    def add_dropout(self, i):
        if self.dropout > 0:
            dp_name = "DROPOUT_" + str(i)
                
            setattr(self, dp_name, Dropout(self.dropout))
            self.layers += [dp_name]            

    def add_activation(self, i):
        a_name = "ACTIVATION_" + str(i)
            
        setattr(self, a_name, getattr(torch.nn, self.activation)())
        self.layers += [a_name]

    def add_conv_layer(self, i, c1, c2, e1=None, e2=None):
        if (i != 0) and (self.layer_type == "GATv2Conv"):
            c1 *= self.heads
            
        if (i == len(self.channels)) and (self.layer_type == "GATv2Conv"):
            heads = 1
        else:
            heads = None
            
        layer = self.build_layer(c1, c2, heads=heads, e1=e1, e2=e2)
        layer_name = f"CONV_{i}"            
        setattr(self, layer_name, layer)
        self.layers += [layer_name]     

    def add_mlp_layer(self, i, n1, n2, z):
        if (i == 0) and (self.layer_type == "GATv2Conv"):
            n1 *= self.heads
        
        layer = Linear(n1, n2)
        layer_name  = f"MLP_{i}_{z}"
        
        setattr(self, layer_name, layer)
        self.regr_layers[z, i].append(layer_name)

    def add_regr_bn_layer(self, i, s, z):
        if self.batch_norm:
            bn_name = "BN_" + str(i) + f"_{z}"
            setattr(self, bn_name, BatchNorm(s))
            self.regr_layers[z, i].append(bn_name)

    def add_regr_dropout(self, i, z):
        if self.dropout > 0:
            dp_name = "DROPOUT_" + str(i) + f"_{z}"
                
            setattr(self, dp_name, Dropout(self.dropout))
            self.regr_layers[z, i].append(dp_name)

    def add_regr_activation(self, i, z):
        a_name = "ACTIVATION_" + str(i) + f"_{z}"
            
        setattr(self, a_name, getattr(torch.nn, self.activation)())
        self.regr_layers[z, i].append(a_name)

    def compute_test_mask(self, batch_size):        
        return np.concatenate(
            [i * self.n_nodes + np.array(self.test_mask)
             for i in range(batch_size)])

    def get_test_mask(self, batch_size):
        if batch_size == self.batch_size:
            return self.regular_test_mask
        else:
            return self.compute_test_mask(batch_size)    
    
        
    def __repr__(self):
        s = "INPUT GRAPH = " + str((self.n_nodes, self.input_shape))
        s += "\nMODEL CONFIG : \n"
        
        previous_conv = None
        for l in self.layers:            
            layer = getattr(self, l)
            if "CONV" in l:
                previous_conv = layer
                
            if l == "POOLING":
                if self.pooling == "flat":
                    str_layer = "SortAggregation"
                else:
                    str_layer = layer.__name__
                    
                to_add = " : " + str(previous_conv.out_channels)
                if str_layer == "SortAggregation":
                    to_add += " * " + str(self.n_nodes)
                if type(previous_conv) == tgnn.GATv2Conv:
                    to_add += " * " + str(self.heads)
                str_layer += to_add
            else:
                str_layer = str(layer)
            s += l + " : " + str_layer + "\n"

        if len(self.regr) == 0:
            return s

        s += "\nFOR ALL " + str(self.n_out_nodes) +  " NODES : \n"
        for i in range(self.regr_layers.shape[1]):
            layers = self.regr_layers[0, i]
            for l in layers:
                layer = getattr(self, l)
                s += l + " : " + str(layer) + "\n"            
        return s

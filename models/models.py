# from models.layers import *
from itertools import combinations
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, TAGConv, GATConv
from models.mlp import MLP
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb

class GNNModel(nn.Module):
    '''
    Docs:
    GNN Model for merchant embedding. Return the merchant vector to represent the location relation in travel time space.

    Network Framework:
        G.x -> Embedding -> G.x
        G -> GNN Model -> x
        edge(v1, v2) -> x -> x1, x2
        cat(x1, x2) -> MLP -> y
        return x
    '''
    def __init__(self, layers, in_features, hidden_features, out_features, prop_depth, dropout=0.0, model_name='DE-GNN', len_distmax = 25, cat_num = 0, cat_embed_dim = 0, cat_features = 0):
        '''
        Input
        ---------
        layers: GNN embedding layers
        in_features: input features dim
        hidden_features: hidden dim
        out_features: output dim
        prop_depth: number of hops for one layer
        dropout: dropout rate
        model_name: model_name for cache
        len_distmax: for PGNN linear layer
        cat_num: types number of categorical feature
        cat_embed_dim: embedding dim for categorical feature
        cat_features: number of categorical feature
        '''
        super(GNNModel, self).__init__()
        self.in_features, self.hidden_features, self.out_features, self.model_name = in_features+cat_features*cat_embed_dim, hidden_features, out_features, model_name
        Layer = self.get_layer_class()
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList()
        self.len_distmax = len_distmax
        self.cat_num = cat_num
        self.cat_embed_dim = cat_embed_dim
        if self.cat_num >0:
            self.embedding = nn.Embedding(self.cat_num, self.cat_embed_dim)
        if self.model_name == 'PGNN':
            self.layers.append(Layer(input_dim=self.in_features, feature_dim=32, hidden_dim=self.hidden_features, output_dim=self.hidden_features,
                 feature_pre=True, layer_num=layers, dropout=dropout))
            self.layers.append(nn.Linear(len_distmax, self.hidden_features))
        else:
            if self.model_name == 'DE-GNN':
                self.layers.append(Layer(in_channels=self.in_features, out_channels=self.hidden_features, K=prop_depth))
            elif self.model_name == 'GIN':
                self.layers.append(
                    Layer(MLP(num_layers=2, input_dim=self.in_features, hidden_dim=self.hidden_features, output_dim=self.hidden_features)))
            else:
                self.layers.append(Layer(in_channels=self.in_features, out_channels=self.hidden_features))
            if layers > 1:
                for i in range(layers - 1):
                    if self.model_name == 'DE-GNN':
                        self.layers.append(Layer(in_channels=self.hidden_features, out_channels=self.hidden_features, K=prop_depth))
                    elif self.model_name == 'GIN':
                        self.layers.append(Layer(MLP(num_layers=2, input_dim=self.hidden_features, hidden_dim=self.hidden_features,
                                                    output_dim=self.hidden_features)))
                    elif self.model_name == 'GAT':
                        self.layers.append(Layer(in_channels=self.hidden_features, out_channels=self.hidden_features, heads=1))
                    else:
                        # for GCN and GraphSAGE
                        self.layers.append(Layer(in_channels=self.hidden_features, out_channels=self.hidden_features))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.hidden_features) for i in range(layers)])
        self.merger = nn.Linear(3 * self.hidden_features, self.hidden_features)
        self.feed_forward = FeedForwardNetwork(self.hidden_features*2, self.out_features)

    def forward(self, batch, train_mask):
        x = batch.x
        if self.cat_num > 0:
            cat_attributes = batch.cat_attributes
            cat_attributes = self.embedding(cat_attributes)
            cat_attributes = cat_attributes.view(cat_attributes.size()[0],-1)
            x = torch.cat([x, cat_attributes], dim = 1)
            # batch.x = x

        edge_index = batch.edge_index
        for i, layer in enumerate(self.layers):
            # edge_weight = None
            # # if not layer.normalize:
            # #     edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
            if self.model_name == 'PGNN':
                if i == 0:
                    x = layer(batch)
                else:
                    x = layer(x)
            else:
                x = layer(x, edge_index)# , edge_weight=None)
            x = self.act(x)
            x = self.dropout(x)  # [n_nodes, mini_batch, input_dim]
            if self.model_name == 'DE-GNN':
                x = self.layer_norms[i](x)
        x = self.get_minibatch_embeddings(x, batch, train_mask)
        x = self.feed_forward(x)
        return x

    def get_minibatch_embeddings(self, x, batch, train_mask):
        edge_index = batch.edge_index
        edge_mask_train = edge_index[:,np.where(train_mask==1)[0]]
        nodes_first = torch.index_select(x, 0, edge_mask_train[0])
        nodes_second = torch.index_select(x, 0, edge_mask_train[1])
        x = torch.cat([nodes_first, nodes_second], dim = 1)
        return x

    def pool(self, x):
        if x.size(1) == 1:
            return torch.squeeze(x, dim=1)
        # use mean/diff/max to pool each set's representations
        x_diff = torch.zeros_like(x[:, 0, :], device=x.device)
        for i, j in combinations(range(x.size(1)), 2):
            x_diff += torch.abs(x[:, i, :]-x[:, j, :])
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x = self.merger(torch.cat([x_diff, x_mean, x_max], dim=-1))
        return x

    def get_layer_class(self):
        layer_dict = {'DE-GNN': TAGConv, 'GIN': GINConv, 'GCN': GCNConv, 'GraphSAGE': SAGEConv, 'GAT': GATConv, 'PGNN':PGNN}  # TAGConv essentially sums up GCN layerwise outputs, can use GCN instead 
        Layer = layer_dict.get(self.model_name)
        if Layer is None:
            raise NotImplementedError('Unknown model name: {}'.format(self.model_name))
        return Layer

    def short_summary(self):
        return 'Model: {}, #layers: {}, in_features: {}, hidden_features: {}, out_features: {}'.format(self.model_name, self.layers, self.in_features, self.hidden_features, self.out_features)


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_features, out_features, act=nn.ReLU(), dropout=0):
        super(FeedForwardNetwork, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features), self.act, self.dropout)
        self.layer2 = nn.Linear(in_features, out_features)

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        return x

class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure

### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=0.0, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout>0:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position
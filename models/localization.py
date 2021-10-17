'''
Adapted from https://github.com/weihua916/powerful-gnns
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return F.log_softmax(output, dim=-1)

#Localization with lienar output
class Localization(nn.Module):
    '''
    Docs:
    Localization Model. Predict travel time between merchants in real time. 

    Network framework:
        traj -> Embedding -> Transformer -> x1
        next shop -> Embedding -> x2
        cat(x1[-1], x2) -> MLP -> y
    '''
    def __init__(self, cat_num, cattype_nums, cat_embed_dim, lstm_input_dim, shop_input_dim, hidden_dim, num_layers, output_dim = 1, lstm_num_layers = 2, dropout=0.0, bidirectional = False):
        '''
        Input:
        ---------
        cat_num: number of categorical features for embedding. 
        cattype_nums: types number of categorical features. 
        cat_embed_dim: dimensionality of categorical feature embedding.
        lstm_input_dim: dimensionality of transformer input features.
        shop_input_dim: dimensionality of next merchant input features.
        hidden_dim: dimensionality of hidden units at ALL layers.
        num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
        output_dim: 1
        lstm_num_layers: number of transformer layers
        dropout: dropout rate
        bidirectional: bidirectional or not
        '''
        super(Localization, self).__init__()

        self.cat_num = cat_num
        self.cattype_nums = cattype_nums
        self.cat_embed_dim = cat_embed_dim
        self.lstm_input_dim = lstm_input_dim
        self.shop_input_dim = shop_input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.criterion = torch.nn.functional.smooth_l1_loss
        self.embedding = nn.Embedding(self.cattype_nums, self.cat_embed_dim, padding_idx=0)
        # self.lstm = nn.GRU(input_size=self.lstm_input_dim, hidden_size=self.hidden_dim, batch_first=True, num_layers = lstm_num_layers, dropout = dropout, bidirectional = bidirectional) 
        self.lstm = TransformerModel(ntoken=lstm_input_dim, ninp=256, nhead=8, nhid=hidden_dim, nlayers=lstm_num_layers)

        self.num_layers = num_layers
        self.shop_linear = nn.Linear(self.shop_input_dim, self.hidden_dim)
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim+self.lstm_input_dim))

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(self.hidden_dim+self.lstm_input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(self.hidden_dim+self.lstm_input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, batch_traj, batch_length, batch_shop):
        # lstm
        cat_traj = self.embedding(batch_traj[:,:,-self.cat_num:].long()).view(batch_traj.size()[0],batch_traj.size()[1],-1)
        traj = torch.cat([batch_traj[:,:,:-self.cat_num], cat_traj], dim = 2)
        traj_out = self.lstm.forward(traj)
        traj_out = traj_out[list(range(traj_out.size()[0])),batch_length-1,:]

        # shop encoder
        cat_shop = self.embedding(batch_shop[:,-self.cat_num:].long()).view(batch_shop.size()[0],-1)
        shop = torch.cat([batch_shop[:,:-self.cat_num], cat_shop], dim = 1)
        shop_out = self.shop_linear(shop)

        # concat
        h = torch.cat([shop_out, traj_out], dim=1)
        h = self.dropout(self.act(self.batch_norms[0](h)))

        # fc
        if self.linear_or_not:
            # If linear model
            out = self.linear(h)
        else:
            # If MLP
            for layer in range(self.num_layers - 1):
                h = self.dropout(self.act(self.batch_norms[layer+1](self.linears[layer](h))))
            out = self.linears[self.num_layers - 1](h)
        return out

    def loss(self, batch_traj, batch_length, batch_shop, label):
        out = self.forward(batch_traj, batch_length, batch_shop)
        loss = self.criterion(out.view(-1), label)
        return loss
import networkx as nx
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from models.models import *
import random
import os
import multiprocessing as mp
from tqdm import tqdm
import time
import sys
from copy import deepcopy
from torch_geometric.data import DataLoader, Data
import torch_geometric.utils as tgu
from debug import *
import torch
import pickle
from geopy.distance import geodesic

def check(args):
    # if args.dataset == 'foodweb' and not args.directed:
    #     raise Warning('dataset foodweb is essentially a directed network but currently treated as undirected')
    # if args.dataset == 'simulation':
    #     if args.n is None:
    #         args.n = [10, 20, 40, 80, 160, 320, 640, 1280]
    #     if args.max_sp < args.T:
    #         raise Warning('maximum shortest path distance (max_sp) is less than max number of layers (T), which may deteriorate model capability')
    pass


def get_device(args):
    gpu = args.gpu
    return torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')


def get_optimizer(model, lr, args):
    optim = args.optimizer
    if optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    elif optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.l2)
    elif optim == 'rprop':
        return torch.optim.Rprop(model.parameters(), lr=lr)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=args.l2)
    elif optim == 'adamax':
        return torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=args.l2)
    else:
        raise NotImplementedError


def estimate_storage(dataloaders, names, logger):
    total_gb = 0
    for dataloader, name in zip(dataloaders, names):
        dataset = dataloader.dataset
        storage = 0
        total_length = len(dataset)
        sample_size = 100
        for i in np.random.choice(total_length, sample_size):
            storage += (sys.getsizeof(dataset[i].x.storage()) + sys.getsizeof(dataset[i].edge_index.storage()) +
                        sys.getsizeof(dataset[i].y.storage())) + sys.getsizeof(dataset[i].set_indices.storage())
        gb = storage*total_length/sample_size/1e9
        total_gb += gb
    logger.info('Data roughly takes {:.4f} GB in total'.format(total_gb))
    return total_gb


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_label(dir, task):
    if task == 'node_classification':
        f_path = dir + 'labels.txt'
        fin_labels = open(f_path)
        labels = []
        node_id_mapping = dict()
        for new_id, line in enumerate(fin_labels.readlines()):
            old_id, label = line.strip().split()
            labels.append(int(label))
            node_id_mapping[old_id] = new_id
        fin_labels.close()
    else:
        labels = None
        nodes = []
        with open(dir + 'edges.txt') as ef:
            for line in ef.readlines():
                nodes.extend(line.strip().split()[:2])
        nodes = sorted(list(set(nodes)))
        node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    return labels, node_id_mapping


def read_edges(dir, node_id_mapping):
    edges = []
    fin_edges = open(dir + 'edges.txt')
    for line in fin_edges.readlines():
        node1, node2 = line.strip().split()[:2]
        edges.append([node_id_mapping[node1], node_id_mapping[node2]])
    fin_edges.close()
    return edges

def merge_data(data_df, time_interval):
    data_df = data_df.drop_duplicates().sort_values(['rider_id_hash','rider_arrival'])
    data_df = data_df[data_df['stay_time']<1800]
    data_df['shop_id_shift'] = data_df.groupby(['rider_id_hash'])['shop_id_hash'].shift(-1)

    data_df['rider_arrival_shift'] = data_df.groupby(['rider_id_hash'])['rider_arrival'].shift(-1)
    data_df['rider_departure_shift'] = data_df.groupby(['rider_id_hash'])['rider_departure'].shift(-1)

    data_df = data_df.reset_index(drop = True)
    tmp = data_df[data_df['shop_id_shift']==data_df['shop_id_hash']]

    tmp['time_interval'] = (pd.to_datetime(tmp['rider_arrival_shift']) - pd.to_datetime(tmp['rider_departure'])).apply(lambda x:x.total_seconds())

    tmp = tmp[tmp['time_interval']<time_interval]
#     print(len(tmp))
    if len(tmp) == 0:
        return data_df, True
    tmp['rider_arrival'] = tmp.apply(lambda x: min(x['rider_arrival'],x['rider_arrival_shift']),axis = 1)
    tmp['rider_departure'] = tmp.apply(lambda x: min(x['rider_departure'],x['rider_departure_shift']),axis = 1)

    tmp['stay_time'] = (pd.to_datetime(tmp['rider_departure']) - pd.to_datetime(tmp['rider_arrival'])).apply(lambda x:x.total_seconds())

    data_df = data_df.drop(index = np.concatenate([np.array(tmp.index), np.array(tmp.index)+1]))

    data_df = pd.concat([data_df, tmp[data_df.columns]])
    return data_df, False

def read_file(args, logger):
    '''
    Docs:
    Data preprocess.
    
    Return
    ---------
    G: merchant graph(networkx.Graph)
    data_df: merchant-merchant DataFrame
    ren_df: encounter(merchant-courier) features DataFrame
    ren_catnums: encounter categorical features count
    train_df, test_df: None
    '''
    encounter_df = pd.read_csv(args.feature_path)
    inshop_df = pd.read_csv(args.inshop_path)
    encounter_df = pd.read_csv(args.feature_path)
    # encounter_df = encounter_df.drop(['label_wifi'], axis = 1)
    encounter_df = pd.merge(left = encounter_df, 
                            right = inshop_df,
                        on = ['event_id','ds'])
    encounter_df = encounter_df[encounter_df['pred']==1]
    # encounter_df = encounter_df.drop(['label_wifi'], axis = 1)
    merchant_cat_features = ['mer_order_mode','mer_first_category_id','mer_manufacturer','mer_brand']
    merchant_num_features = ['mer_is_bpc','mer_avg_order_price','mer_avg_merchant_confirmation_time',
                            'mer_avg_order_delivery_time','mer_avg_courier_waiting_time','mer_avg_predict_cooking_time',
                            'mer_is_sia','mer_is_chain','mer_is_premium']
    ren_cat_features = ['env_phone_hour','env_week','env_weather','mer_order_mode','mer_first_category_id','mer_manufacturer','mer_brand']
    ren_num_features = ['auto_confirm_order','mer_busy_state','mer_bluetooth_on','mer_is_bpc','mer_avg_order_price',
                        'mer_avg_merchant_confirmation_time','mer_avg_order_delivery_time','mer_avg_courier_waiting_time','mer_avg_predict_cooking_time',
                        'mer_order_cnt_this_hour','mer_is_sia','mer_is_chain','cou_experience_days','cou_order_cnt_this_hour',
                        'mer_last_confirm_order_datetime','mer_last_print_order_datetime','dist',
                        'ren_rssi_avg', 'ren_rssi_min', 'ren_rssi_max','ren_rssi_var', 'ren_data_cnt']

    # 获取骑手序列beacon_df1
    tmp = encounter_df[['shop_id_hash','mer_shop_latitude','mer_shop_longitude']].drop_duplicates().dropna()
    encounter_df = pd.merge(left = encounter_df.drop(['mer_shop_latitude','mer_shop_longitude'], axis = 1)
                            , right = tmp, on = 'shop_id_hash', how = 'left')
    encounter_df.loc[(~encounter_df['mer_na_latitude_gps'].isna())&(~encounter_df['mer_na_longitude_gps'].isna()),
                'dist'] = encounter_df[
        (~encounter_df['mer_na_latitude_gps'].isna())&(~encounter_df['mer_na_longitude_gps'].isna())].apply(
        lambda x: geodesic((x.mer_shop_latitude, x.mer_shop_longitude),(x.mer_na_latitude_gps, x.mer_na_longitude_gps)).km, axis = 1)
    encounter_df[['auto_confirm_order','mer_busy_state','mer_bluetooth_on','mer_is_bpc','mer_is_sia',
        'mer_is_chain','mer_is_premium','mer_is_zps']] = encounter_df[['auto_confirm_order','mer_busy_state','mer_bluetooth_on','mer_is_bpc','mer_is_sia',
                                                                    'mer_is_chain','mer_is_premium','mer_is_zps']].astype(bool).astype(int)
    encounter_df['env_phone_time'] = pd.to_datetime(encounter_df['env_phone_time'])
    encounter_df['mer_last_print_order_datetime'] = pd.to_datetime(encounter_df['mer_last_print_order_datetime'])
    encounter_df['mer_last_confirm_order_datetime'] = pd.to_datetime(encounter_df['mer_last_confirm_order_datetime'])
    encounter_df['mer_last_print_order_datetime'] = (encounter_df['env_phone_time']-encounter_df['mer_last_print_order_datetime']).apply(lambda x:x.total_seconds())
    encounter_df['mer_last_confirm_order_datetime'] = (encounter_df['env_phone_time']-encounter_df['mer_last_confirm_order_datetime']).apply(lambda x:x.total_seconds())
    ren_df = encounter_df[ren_cat_features+ren_num_features+['ds','event_id','label_wifi']]

    ren_df1 = encounter_df[
        ['rider_id_hash', 'shop_id_hash', 'ren_detected_at_min','ren_detected_at_max','event_id', 'ds']
        ].rename(columns = {'ren_detected_at_min':'rider_arrival','ren_detected_at_max':'rider_departure'}).drop_duplicates()

    ren_df1['stay_time'] = (pd.to_datetime(ren_df1['rider_departure']) - pd.to_datetime(ren_df1['rider_arrival'])).apply(lambda x:x.total_seconds())

    while True:
        ren_df1, tag = merge_data(ren_df1, time_interval = args.TIME_INTERVAL)
        if tag:
            break
    ren_df = pd.merge(left = ren_df1, right = ren_df, on = ['ds','event_id'], how = 'left')

    ren_num_features += ['stay_time']
    ren_df[ren_num_features] = ren_df[ren_num_features].apply(lambda x: x.fillna(x.mean()))
    ren_df[ren_cat_features] = ren_df[ren_cat_features].fillna('unknown')
    ren_df[ren_cat_features] = ren_df[ren_cat_features].astype(str)
    le = LabelEncoder()
    ren_df[ren_cat_features] = le.fit_transform(ren_df[ren_cat_features].values.reshape(-1,)).reshape(-1,len(ren_cat_features))
    ren_df[ren_cat_features] += 1
    ren_catnums = len(le.classes_)+2
    nodes = sorted(ren_df['shop_id_hash'].unique().tolist())
    node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(nodes)}
    rider_ids = sorted(ren_df['rider_id_hash'].unique().tolist())
    rider_id_mapping = {old_id: new_id for new_id, old_id in enumerate(rider_ids)}
    ren_df['shop_id'] = ren_df.apply(lambda x: node_id_mapping[x.shop_id_hash], axis = 1)
    ren_df['rider_id'] = ren_df.apply(lambda x: rider_id_mapping[x.rider_id_hash], axis = 1)

    event_id = ren_df[['event_id','ds']].drop_duplicates().reset_index(drop = True).reset_index()
    event_id['index'] += 1
    ren_df = pd.merge(left = ren_df, right = event_id, on = ['event_id','ds']).drop('event_id', axis = 1).rename(columns = {'index':'event_id'})
    # 获取商户连边data_df
    data_df = ren_df[['shop_id','rider_arrival','rider_departure','rider_id']].drop_duplicates()
    data_df = data_df.sort_values(['rider_id','rider_arrival'])
    data_df['rider_departure_shift'] = data_df.groupby(['rider_id'])['rider_departure'].shift(1)
    data_df['shop_id2'] = data_df.groupby(['rider_id'])['shop_id'].shift(1)
    data_df['time_diff'] = (pd.to_datetime(data_df['rider_arrival'])-pd.to_datetime(data_df['rider_departure_shift'])).apply(lambda x:x.total_seconds())
    data_df.loc[data_df['time_diff']<0,'time_diff'] = 0

    data_df['shop_id1'] = data_df['shop_id']

    data_df = data_df[data_df['time_diff']<args.THRE]

    data_df = data_df.groupby(['shop_id1','shop_id2'])['time_diff'].describe().reset_index()
    data_df[['shop_id1','shop_id2']] = data_df[['shop_id1','shop_id2']].astype(int)

    edges = list(zip(data_df['shop_id1'],data_df['shop_id2']))
    edges = list(map(list, edges))

    if not args.directed:
        G = nx.Graph(edges)
    else:
        G = nx.DiGraph(edges)

    # 解决有些node没出现的问题
    node_id_mapping_new = list(sorted(nx.connected_components(G))[0])
    if len(node_id_mapping_new) < max(node_id_mapping.values()):
        node_id_mapping_df = pd.DataFrame({'shop_id':list(range(len(node_id_mapping_new))), 'old': node_id_mapping_new})
        node_id_mapping_new = {old_id: new_id for new_id, old_id in enumerate(node_id_mapping_new)}

        node_id_mapping_dict = {}
        for k in node_id_mapping.keys():
            if node_id_mapping_new.get(node_id_mapping[k]) is not None:
                node_id_mapping_dict[k] = node_id_mapping_new.get(node_id_mapping[k])

        data_df = (pd.merge(left = data_df, 
                        right = node_id_mapping_df.rename(columns = {'old':'shop_id1'}), 
                        on = 'shop_id1')).drop('shop_id1',axis = 1).rename(columns = {'shop_id':'shop_id1'})
        data_df = (pd.merge(left = data_df, 
                        right = node_id_mapping_df.rename(columns = {'old':'shop_id2'}), 
                        on = 'shop_id2')).drop('shop_id2',axis = 1).rename(columns = {'shop_id':'shop_id2'})
        edges = list(zip(data_df['shop_id1'],data_df['shop_id2']))
        edges = list(map(list, edges))
        G = nx.Graph(edges)
        node_id_mapping = node_id_mapping_dict
        ren_df = pd.merge(left = ren_df.rename(columns={'shop_id':'old'}), 
                    right = node_id_mapping_df, on = 'old').drop('old',axis = 1)

    attributes = np.zeros((G.number_of_nodes(), 1), dtype=np.float32)
    if args.use_degree:
        attributes += np.expand_dims(np.log(get_degrees(G)+1), 1).astype(np.float32)
    if args.use_attributes:
        encounter_df = encounter_df[['shop_id_hash']+merchant_cat_features+merchant_num_features].drop_duplicates()
        encounter_df = encounter_df.groupby('shop_id_hash').max().reset_index()
        encounter_df = encounter_df[encounter_df['shop_id_hash'].isin(node_id_mapping.keys())]
        shop_id = pd.DataFrame({'shop_id':list(range(G.number_of_nodes()))})
        encounter_df['shop_id'] = encounter_df.apply(lambda x: node_id_mapping[x.shop_id_hash], axis = 1)
        encounter_df = pd.merge(left = encounter_df, right = shop_id, on = 'shop_id', how = 'right')
        encounter_df[merchant_num_features] = encounter_df[merchant_num_features].apply(lambda x: x.fillna(x.mean()))
        encounter_df[merchant_cat_features] = encounter_df[merchant_cat_features].fillna('unknown')
        encounter_df[merchant_cat_features] = encounter_df[merchant_cat_features].astype(str)
        # le = LabelEncoder()
        encounter_df[merchant_cat_features] = le.transform(encounter_df[merchant_cat_features].values.reshape(-1,)).reshape(-1,len(merchant_cat_features))
        assert(G.number_of_nodes()==encounter_df.shape[0]), 'cat_attributes shape error'
        encounter_df = encounter_df.sort_values('shop_id')
        cat_attributes = encounter_df[merchant_cat_features].values
        G.graph['cat_attributes'] = cat_attributes
        num_attributes = encounter_df[merchant_num_features].values
        attributes = np.concatenate([attributes, num_attributes],axis = 1) 

    cache_path = os.path.join(args.cache_dir,"{}_{}_{}.csv".format('idmaps',args.version_name,args.model))
    with open(cache_path,'wb') as f:
        pickle.dump((node_id_mapping, rider_id_mapping, event_id, le), f)

    G.graph['attributes'] = attributes
    logger.info('number of nodes: {}, number of edges: {}. Directed: {}'.format(G.number_of_nodes(),G.number_of_edges(),args.directed))
    test_df = None
    train_df = None
    return G, data_df, ren_df, train_df, test_df, ren_catnums


def get_gnn_data(G, data_df, args, logger):
    '''
    Docs:
    Graph embedding data preprocess. method: Link prediction.

    Return
    ---------
    data(torch_geometric.data.Data) including train, val, test indices
    '''
    G = deepcopy(G)  # to make sure original G is unchanged
    sp_flag = 'sp' in args.feature
    rw_flag = 'rw' in args.feature
    feature_flags = (sp_flag, rw_flag)
    
    G, labels, set_indices, (train_mask, val_mask, test_mask) = generate_samples_labels_graph(G, data_df, args, logger)

    data = extract_subgaphs(G, labels, set_indices, 
                                 feature_flags=feature_flags, 
                                 max_sprw=(args.max_sp, args.rw_depth), logger=logger, args = args, debug=args.debug)

    if args.model == 'PGNN':
        dists_removed = precompute_dist_data(data.edge_index, data.num_nodes,
                                                        approximate=args.approximate)
        data.dists = torch.from_numpy(dists_removed).float()
        preselect_anchor(data, layer_num=args.layers, anchor_num=args.anchor_num)

    data.train_mask = np.array(np.concatenate([train_mask,np.zeros(set_indices.shape[0])]),dtype = int)
    data.val_mask = np.array(np.concatenate([val_mask,np.zeros(set_indices.shape[0])]),dtype = int)
    data.test_mask = np.array(np.concatenate([test_mask,np.zeros(set_indices.shape[0])]),dtype = int)

    logger.info('Train size :{}, val size: {}, val ratio: {}'.format(np.sum(train_mask), np.sum(val_mask), args.test_ratio))
    return data


def generate_samples_labels_graph(G, data_df, args, logger):
    pos_edges = np.array(data_df[['shop_id1','shop_id2']], dtype=np.int32)
    labels = data_df['25%'].values

    n_pos_edges = pos_edges.shape[0]

    pos_test_size = int(args.test_ratio * n_pos_edges)

    test_pos_indices = random.sample(range(n_pos_edges), pos_test_size)

    test_mask = get_mask(test_pos_indices, length=n_pos_edges)
    train_mask = np.ones_like(test_mask) - test_mask
    # G.remove_edges_from(pos_edges[test_pos_indices].tolist())

    val_mask = np.zeros(n_pos_edges)
    val_mask[random.sample(list(np.where(test_mask==1)[0]),int(0.5*pos_test_size))] = 1
    val_mask = np.array(val_mask, dtype = int)
    test_mask -= val_mask

    permutation = np.random.permutation(n_pos_edges)
    set_indices = pos_edges[permutation]
    labels = labels[permutation]
    train_mask, val_mask, test_mask = train_mask[permutation], val_mask[permutation], test_mask[permutation]

    logger.info('Generate {} train+val+test instances in total.'.format(set_indices.shape[0]))
    return G, labels, set_indices, (train_mask, val_mask, test_mask)


def extract_subgaphs(G, labels, set_indices, feature_flags, max_sprw, logger, args, debug=False):
    # deal with adj and features,每个sample对应一个子图对应1个label，得到data_list
    logger.info('Encode positions')
    sp_flag, rw_flag = feature_flags
    max_sp, rw_depth = max_sprw

    edge_index = torch.tensor(set_indices).long().t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0], ]], dim=-1)

    # Construct x from x_list
    x_list = []
    attributes = G.graph['attributes']
    if attributes is not None:
        new_attributes = torch.tensor(attributes, dtype=torch.float32)
        if new_attributes.dim() < 2:
            new_attributes.unsqueeze_(1)
        x_list.append(new_attributes)
    # if deg_flag:
    #     x_list.append(torch.log(tgu.degree(new_edge_index[0], num_nodes=num_nodes, dtype=torch.float32).unsqueeze(1)+1))
    if sp_flag:
        features_sp_sample = get_features_sp_sample(G,  np.array(G.nodes), max_sp=max_sp)
        features_sp_sample = torch.from_numpy(features_sp_sample).float()
        x_list.append(features_sp_sample)
    if rw_flag:
        adj = np.asarray(nx.adjacency_matrix(G, nodelist=np.arange(G.number_of_nodes(), dtype=np.int32)).todense().astype(np.float32))  # [n_nodes, n_nodes]
        features_rw_sample = get_features_rw_sample(adj,  np.array(G.nodes), rw_depth=rw_depth)
        features_rw_sample = torch.from_numpy(features_rw_sample).float()
        x_list.append(features_rw_sample)

    x = torch.cat(x_list, dim=-1)
    y = torch.tensor(labels, dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, y=y)
    if args.use_attributes:
        data.cat_attributes = torch.tensor(G.graph['cat_attributes'], dtype = torch.long)
    return data


def get_gnn_model(layers, in_features, out_features, prop_depth, args, logger, len_distmax, cat_num, cat_features):
    '''
    Docs:
    Get GNN model.

    Input
    ---------
    layers: GNN embedding layers
    in_features: input features dim
    out_features: output dim
    prop_depth: number of hops for one layer
    args:
    logger:
    len_distmax: for PGNN linear layer
    cat_num: types number of categorical feature
    cat_features: number of categorical feature

    Return
    ---------
    GNN model
    '''
    model_name = args.model
    if model_name in ['DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT', 'PGNN']:
        model = GNNModel(layers=layers, in_features=in_features, hidden_features=args.hidden_features,
                         out_features=out_features, prop_depth=prop_depth, dropout=args.dropout,
                         model_name=model_name, len_distmax = len_distmax, cat_num = cat_num, cat_embed_dim = args.cat_embed_dim, cat_features = cat_features)
    else:
        return NotImplementedError
    logger.info(model.short_summary())
    return model


def get_features_sp_sample(G, node_set, max_sp):
    dim = max_sp + 2
    set_size = len(node_set)
    sp_length = np.ones((G.number_of_nodes(), set_size), dtype=np.int32) * -1
    for i, node in enumerate(node_set):
        for node_ngh, length in nx.shortest_path_length(G, source=node).items():
            sp_length[node_ngh, i] = length
    sp_length = np.minimum(sp_length, max_sp)
    onehot_encoding = np.eye(dim, dtype=np.float64)  # [n_features, n_features]
    features_sp = onehot_encoding[sp_length].sum(axis=1)
    return features_sp


def get_features_rw_sample(adj, node_set, rw_depth):
    epsilon = 1e-6
    adj = adj / (adj.sum(1, keepdims=True) + epsilon)
    rw_list = [np.identity(adj.shape[0])[node_set]]
    for _ in range(rw_depth):
        rw = np.matmul(rw_list[-1], adj)
        rw_list.append(rw)
    features_rw_tmp = np.stack(rw_list, axis=2)  # shape [set_size, N, F]
    # pooling
    features_rw = features_rw_tmp.sum(axis=0)
    return features_rw


def get_hop_num(prop_depth, layers, max_sprw, feature_flags):
    # TODO: may later use more rw_depth to control as well?
    return int(prop_depth * layers) + 1   # in order to get the correct degree normalization for the subgraph


def shortest_path_length(graph):
    sp_length = np.ones([graph.number_of_nodes(), graph.number_of_nodes()], dtype=np.int32) * -1
    for node1, value in nx.shortest_path_length(graph):
        for node2, length in value.items():
            sp_length[node1][node2] = length

    return sp_length


def split_dataset(n_samples, test_ratio, stratify=None):
    train_indices, test_indices = train_test_split(list(range(n_samples)), test_size=test_ratio, stratify=stratify)
    train_mask = get_mask(train_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)
    return train_mask, test_mask


def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.int8)

def sample_neg_sets(G, n_samples, set_size):
    neg_sets = []
    n_nodes = G.number_of_nodes()
    max_iter = 1e9
    count = 0
    while len(neg_sets) < n_samples:
        count += 1
        if count > max_iter:
            raise Exception('Reach max sampling number of {}, input graph density too high'.format(max_iter))
        candid_set = [int(random.random() * n_nodes) for _ in range(set_size)]
        for node1, node2 in combinations(candid_set, 2):
            if not G.has_edge(node1, node2):
                neg_sets.append(candid_set)
                break

    return neg_sets


def collect_tri_sets(G):
    tri_sets = set(frozenset([node1, node2, node3]) for node1 in G for node2, node3 in combinations(G.neighbors(node1), 2) if G.has_edge(node2, node3))
    return [list(tri_set) for tri_set in tri_sets]


def retain_partial(indices, ratio):
    sample_i = np.random.choice(indices.shape[0], int(ratio * indices.shape[0]), replace=False)
    return indices[sample_i], sample_i


def pagerank_inverse(adj, alpha=0.90):
    adj /= (adj.sum(axis=-1, keepdims=True) + 1e-12)
    return np.linalg.inv(np.eye(adj.shape[0]) - alpha * np.transpose(adj, axes=(0,1)))


def split_datalist(data_list, masks):
    # generate train_set
    train_mask, val_test_mask = masks
    num_graphs = len(data_list)
    assert((train_mask.sum()+val_test_mask.sum()).astype(np.int32) == num_graphs)
    assert(train_mask.shape[0] == num_graphs)
    train_indices = np.arange(num_graphs)[train_mask.astype(bool)]
    train_set = [data_list[i] for i in train_indices]
    # generate val_set and test_set
    val_test_indices = np.arange(num_graphs)[val_test_mask.astype(bool)]
    val_test_labels = np.array([data.y for data in data_list], dtype=np.int32)[val_test_indices]
    val_indices, test_indices = train_test_split(val_test_indices, test_size=int(0.5*len(val_test_indices)))
    val_set = [data_list[i] for i in val_indices]
    test_set = [data_list[i] for i in test_indices]
    return train_set, val_set, test_set


def load_datasets(train_set, val_set, test_set, bs):
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=bs, shuffle=True, pin_memory=True, num_workers=num_workers)
    return train_loader, val_loader, test_loader

def split_indices(num_graphs, test_ratio, stratify=None):
    test_size = int(num_graphs*test_ratio)
    val_size = test_size
    train_val_set, test_set = train_test_split(np.arange(num_graphs), test_size=test_size, shuffle=True, stratify=stratify)
    train_set, val_set = train_test_split(train_val_set, test_size=val_size, shuffle=True, stratify=stratify[train_val_set])
    return train_set, val_set, test_set


def get_degrees(G):
    num_nodes = G.number_of_nodes()
    return np.array([dict(G.degree).get(i,0) for i in range(num_nodes)])


# ================================== (obsolete) Just for PGNN =================================================
# Adapted from https://github.com/JiaxuanYou/P-GNN
# # approximate
def get_edge_mask_link_negative_approximate(mask_link_positive, num_nodes, num_negtive_edges):
    links_temp = np.zeros((num_nodes, num_nodes)) + np.identity(num_nodes)
    mask_link_positive = duplicate_edges(mask_link_positive)
    links_temp[mask_link_positive[0],mask_link_positive[1]] = 1
    # add random noise
    links_temp += np.random.rand(num_nodes,num_nodes)
    prob = num_negtive_edges / (num_nodes*num_nodes-mask_link_positive.shape[1])
    mask_link_negative = np.stack(np.nonzero(links_temp<prob))
    return mask_link_negative


# exact version, slower
def get_edge_mask_link_negative(mask_link_positive, num_nodes, num_negtive_edges):
    mask_link_positive_set = []
    for i in range(mask_link_positive.shape[1]):
        mask_link_positive_set.append(tuple(mask_link_positive[:,i]))
    mask_link_positive_set = set(mask_link_positive_set)

    mask_link_negative = np.zeros((2,num_negtive_edges), dtype=mask_link_positive.dtype)
    for i in range(num_negtive_edges):
        while True:
            mask_temp = tuple(np.random.choice(num_nodes,size=(2,),replace=False))
            if mask_temp not in mask_link_positive_set:
                mask_link_negative[:,i] = mask_temp
                break

    return mask_link_negative

def resample_edge_mask_link_negative(data):
    data.mask_link_negative_train = get_edge_mask_link_negative(data.mask_link_positive_train, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_train.shape[1])
    data.mask_link_negative_val = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                      num_negtive_edges=data.mask_link_positive_val.shape[1])
    data.mask_link_negative_test = get_edge_mask_link_negative(data.mask_link_positive, num_nodes=data.num_nodes,
                                                     num_negtive_edges=data.mask_link_positive_test.shape[1])


def deduplicate_edges(edges):
    edges_new = np.zeros((2,edges.shape[1]//2), dtype=int)
    # add none self edge
    j = 0
    skip_node = {} # node already put into result
    for i in range(edges.shape[1]):
        if edges[0,i]<edges[1,i]:
            edges_new[:,j] = edges[:,i]
            j += 1
        elif edges[0,i]==edges[1,i] and edges[0,i] not in skip_node:
            edges_new[:,j] = edges[:,i]
            skip_node.add(edges[0,i])
            j += 1

    return edges_new

def duplicate_edges(edges):
    return np.concatenate((edges, edges[::-1,:]), axis=-1)


# each node at least remain in the new graph
def split_edges(edges, remove_ratio, connected=False):
    e = edges.shape[1]
    edges = edges[:, np.random.permutation(e)]
    if connected:
        unique, counts = np.unique(edges, return_counts=True)
        node_count = dict(zip(unique, counts))

        index_train = []
        index_val = []
        for i in range(e):
            node1 = edges[0,i]
            node2 = edges[1,i]
            if node_count[node1]>1 and node_count[node2]>1: # if degree>1
                index_val.append(i)
                node_count[node1] -= 1
                node_count[node2] -= 1
                if len(index_val) == int(e * remove_ratio):
                    break
            else:
                index_train.append(i)
        index_train = index_train + list(range(i + 1, e))
        index_test = index_val[:len(index_val)//2]
        index_val = index_val[len(index_val)//2:]

        edges_train = edges[:, index_train]
        edges_val = edges[:, index_val]
        edges_test = edges[:, index_test]
    else:
        split1 = int((1-remove_ratio)*e)
        split2 = int((1-remove_ratio/2)*e)
        edges_train = edges[:,:split1]
        edges_val = edges[:,split1:split2]
        edges_test = edges[:,split2:]

    return edges_train, edges_val, edges_test




def edge_to_set(edges):
    edge_set = []
    for i in range(edges.shape[1]):
        edge_set.append(tuple(edges[:, i]))
    edge_set = set(edge_set)
    return edge_set


def get_link_mask(data, remove_ratio=0.2, resplit=True, infer_link_positive=True):
    if resplit:
        if infer_link_positive:
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train, data.mask_link_positive_val, data.mask_link_positive_test = \
            split_edges(data.mask_link_positive, remove_ratio)
    resample_edge_mask_link_negative(data)


def add_nx_graph(data):
    G = nx.Graph()
    edge_numpy = data.edge_index.numpy()
    edge_list = []
    for i in range(data.num_edges):
        edge_list.append(tuple(edge_numpy[:, i]))
    G.add_edges_from(edge_list)
    data.G = G

def single_source_shortest_path_length_range(graph, node_range, cutoff):
    dists_dict = {}
    for node in node_range:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
    return dists_dict

def merge_dicts(dicts):
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def all_pairs_shortest_path_length_parallel(graph,cutoff=None,num_workers=4):
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    if len(nodes)<50:
        num_workers = int(num_workers/4)
    elif len(nodes)<400:
        num_workers = int(num_workers/2)

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(single_source_shortest_path_length_range,
            args=(graph, nodes[int(len(nodes)/num_workers*i):int(len(nodes)/num_workers*(i+1))], cutoff)) for i in range(num_workers)]
    output = [p.get() for p in results]
    dists_dict = merge_dicts(output)
    pool.close()
    pool.join()
    return dists_dict


def precompute_dist_data(edge_index, num_nodes, approximate=0):
        '''
        Here dist is 1/real_dist, higher actually means closer, 0 means disconnected
        :return:
        '''
        graph = nx.Graph()
        edge_list = edge_index.transpose(1,0).tolist()
        graph.add_edges_from(edge_list)

        n = num_nodes
        dists_array = np.zeros((n, n))
        # dists_dict = nx.all_pairs_shortest_path_length(graph,cutoff=approximate if approximate>0 else None)
        # dists_dict = {c[0]: c[1] for c in dists_dict}
        dists_dict = all_pairs_shortest_path_length_parallel(graph,cutoff=approximate if approximate>0 else None)
        for i, node_i in enumerate(graph.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist!=-1:
                    # dists_array[i, j] = 1 / (dist + 1)
                    dists_array[node_i, node_j] = 1 / (dist + 1)
        return dists_array



def get_random_anchorset(n,c=0.5):
    m = int(np.log2(n))
    copy = int(c*m)
    anchorset_id = []
    for i in range(m):
        anchor_size = int(n/np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(n,size=anchor_size,replace=False))
    return anchorset_id

def get_dist_max(anchorset_id, dist, device):
    dist_max = torch.zeros((dist.shape[0],len(anchorset_id))).to(device)
    dist_argmax = torch.zeros((dist.shape[0],len(anchorset_id))).long().to(device)
    for i in range(len(anchorset_id)):
        temp_id = torch.as_tensor(anchorset_id[i], dtype=torch.long)
        dist_temp = dist[:, temp_id]
        dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
        dist_max[:,i] = dist_max_temp
        dist_argmax[:,i] = temp_id[dist_argmax_temp]
    return dist_max, dist_argmax


def preselect_anchor(data, layer_num=1, anchor_num=32, anchor_size_num=4, device='cpu'):

    data.anchor_size_num = anchor_size_num
    data.anchor_set = []
    anchor_num_per_size = anchor_num//anchor_size_num
    for i in range(anchor_size_num):
        anchor_size = 2**(i+1)-1
        anchors = np.random.choice(data.num_nodes, size=(layer_num,anchor_num_per_size,anchor_size), replace=True)
        data.anchor_set.append(anchors)
    data.anchor_set_indicator = np.zeros((layer_num, anchor_num, data.num_nodes), dtype=int)

    anchorset_id = get_random_anchorset(data.num_nodes,c=1)
    data.dists_max, data.dists_argmax = get_dist_max(anchorset_id, data.dists, device)

def gnn_result(gnn_model, data, args):
    '''
    Docs:
    Get merchant embedding result based on GNN model

    Return
    -----------
    merchant matrix
    '''
    device = get_device(args)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()
    with torch.no_grad():
        batch = data.to(device)
        x = batch.x
        if gnn_model.cat_num > 0:
            cat_attributes = batch.cat_attributes
            cat_attributes = gnn_model.embedding(cat_attributes)
            cat_attributes = cat_attributes.view(cat_attributes.size()[0],-1)
            x = torch.cat([x, cat_attributes], dim = 1)
            # batch.x = x

        edge_index = batch.edge_index
        for i, layer in enumerate(gnn_model.layers):
            # edge_weight = None
            # # if not layer.normalize:
            # #     edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
            if gnn_model.model_name == 'PGNN':
                if i == 0:
                    x = layer(batch)
                else:
                    x = layer(x)
            else:
                x = layer(x, edge_index)# , edge_weight=None)
            x = gnn_model.act(x)
            x = gnn_model.dropout(x)  # [n_nodes, mini_batch, input_dim]
            if gnn_model.model_name == 'DE-GNN':
                x = gnn_model.layer_norms[i](x)
    x = x.cpu().numpy()
    return x

def get_traj(tmp, thre):
    # tmp = tmp.sort_values('rider_arrival')
    result_ls = []
    traj_ls = []
    t = 0
    for i in range(len(tmp)):
        if tmp.iloc[i]['time_diff'] < thre:
            t += tmp.iloc[i]['time_diff']
            traj_ls.append((tmp.iloc[i]['event_id'], t))
        else:
            traj_ls.append((tmp.iloc[i]['event_id'], t))
            result_ls.append(traj_ls)
            traj_ls = []
            t = 0
    result_all = []
    label_ls = []
    event_ls = []
    for x in result_ls:
        label_T = 0
        for i in range(len(x)):
            x_tmp = x[i:]
            
            if len(x_tmp) > 1:
                event_ls.append(x_tmp[0][0])
                label_ls.append(x_tmp[0][1]-label_T)
                label_T = x_tmp[0][1]
                T = x_tmp[0][1]
                result = []
                t = 0
                for val in x_tmp[1:]:
                    result.append((val[0], t))
                    t = val[1]-T
                result_all.append(result[::-1])
    df = pd.DataFrame({'traj':result_all, 'label':label_ls, 'event_id_2': event_ls})
    return df

def get_lstm_data(ren_df, shop_matrix, test_df, args):
    '''
    Docs:
    Sequence data preprocess.
    irregular time series. 
    encoder sample:
    original data: (m1,t1),(m2,t2),(m3,t3), (m4,t4); t1<t2<t3<t4 
    generate samples:
    * (m1, t3-t1), (m2, t3-t2), (m3,0), m4 -> t4-t3
    * (m1, t2-t1), (m2, 0), m3 -> t3-t2
    * (m1, 0), m2 -> t2-t1

    Input
    --------
    ren_df: encounter feature DataFrame
    shop_matrix: merchant embedding result based on GNN model
    test_df: test data
    args

    Return
    --------
    dataloaders: (train_loader, val_loader, test_loader); 
        trainset including train_traj(time series), train_lengths(time series steps), train_shop(next merchant), train_label(label: travel time)
    traj_dim: sequence feature dim
    shop_dim: merchant embedding dim
    test_df: test data
    '''
    def traj_matrix(tmp):
        traj_val = []
        for val in tmp:
            traj_val.append(np.concatenate([[val[1]], event_matrix[val[0]]]).reshape(1, 1,-1))
        traj_val = np.concatenate(traj_val,axis = 1)
        traj_val = np.concatenate([traj_val,np.zeros((1, args.range-traj_val.shape[1], traj_val.shape[2]))], axis = 1)
        return traj_val
    # 获取序列
    traj_df = ren_df[['rider_id','event_id','rider_arrival','rider_departure','ds']]
    traj_df = traj_df.sort_values(['rider_id','rider_arrival'],ascending = False)
    traj_df['rider_departure_shift'] = traj_df.groupby('rider_id')['rider_departure'].shift(-1)
    traj_df['time_diff'] = (pd.to_datetime(traj_df['rider_arrival']) - pd.to_datetime(traj_df['rider_departure_shift'])).apply(lambda x: x.total_seconds())
    traj_df.loc[traj_df['time_diff']<0, 'time_diff'] = 0
    # traj_df = traj_df.sort_values(['rider_id','rider_arrival'],ascending = False)
    traj_df = traj_df.groupby(['rider_id','ds']).apply(lambda x: get_traj(x, args.THRE)).reset_index().drop('level_2',axis = 1)
    traj_df['event_id_1'] = traj_df['traj'].apply(lambda x: x[-1][0])

    # 整理event features
    ren_cat_features = ['env_phone_hour','env_week','env_weather','mer_order_mode','mer_first_category_id',
                    'mer_manufacturer','mer_brand']
    ren_num_features = ['auto_confirm_order','mer_busy_state','mer_bluetooth_on','mer_is_bpc','mer_avg_order_price',
                            'mer_avg_merchant_confirmation_time','mer_avg_order_delivery_time',
                        'mer_avg_courier_waiting_time',
                        'mer_avg_predict_cooking_time','mer_order_cnt_this_hour','mer_is_sia','mer_is_chain',
                        'cou_experience_days',
                        'cou_order_cnt_this_hour','mer_last_confirm_order_datetime','mer_last_print_order_datetime'
                    ,'dist','ren_rssi_avg', 'ren_rssi_min', 'ren_rssi_max','ren_rssi_var', 'ren_data_cnt', 'stay_time']
    env_cat_features = ['env_phone_hour','env_week','env_weather','mer_order_mode','mer_first_category_id',
                    'mer_manufacturer','mer_brand']
    env_num_features = ['auto_confirm_order','mer_busy_state','mer_bluetooth_on','mer_is_bpc','mer_avg_order_price',
                            'mer_avg_merchant_confirmation_time','mer_avg_order_delivery_time',
                        'mer_avg_courier_waiting_time',
                        'mer_avg_predict_cooking_time','mer_order_cnt_this_hour','mer_is_sia','mer_is_chain',
                        'cou_experience_days',
                        'cou_order_cnt_this_hour','mer_last_confirm_order_datetime','mer_last_print_order_datetime']

    helper = pd.DataFrame({'event_id':range(ren_df['event_id'].max()+1)})
    ren_df = pd.merge(left = ren_df, right = helper, on = 'event_id', how = 'right')
    ren_df = ren_df.sort_values('event_id')
    ren_df = ren_df.fillna(0)
    event_matrix = np.concatenate([shop_matrix[ren_df['shop_id'].astype(int).values], 
                                ren_df[ren_num_features+ren_cat_features].values],axis = 1)
    env_matrix = np.concatenate([shop_matrix[ren_df['shop_id'].astype(int).values], 
                                ren_df[env_num_features+env_cat_features].values],axis = 1)

    traj_df['traj'] = traj_df['traj'].apply(lambda x: x[-args.range:] if len(x) > args.range else x)
    traj_df['l'] = traj_df['traj'].apply(lambda x:len(x))

    # 得到训练测试集
    train_df = traj_df[traj_df['ds']<args.test_date]
    test_df = traj_df[traj_df['ds']>=args.test_date]
    train_df['traj_matrix'] = train_df['traj'].apply(lambda x: traj_matrix(x))
    test_df['traj_matrix'] = test_df['traj'].apply(lambda x: traj_matrix(x))
    train_df, val_df = train_test_split(train_df, test_size = 0.3)
    train_traj = np.concatenate(train_df['traj_matrix'].values.tolist())
    train_lengths = train_df['l'].values
    train_shop = env_matrix[train_df['event_id_2'].astype(int).values]
    train_label = train_df['label'].values
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(train_traj),
                                         torch.LongTensor(train_lengths),
                                         torch.FloatTensor(train_shop),
                                         torch.FloatTensor(train_label))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    val_traj = np.concatenate(val_df['traj_matrix'].values.tolist())
    val_lengths = val_df['l'].values
    val_shop = env_matrix[val_df['event_id_2'].astype(int).values]
    val_label = val_df['label'].values
    valset = torch.utils.data.TensorDataset(torch.FloatTensor(val_traj),
                                         torch.LongTensor(val_lengths),
                                         torch.FloatTensor(val_shop),
                                         torch.FloatTensor(val_label))
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=True)

    test_traj = np.concatenate(test_df['traj_matrix'].values.tolist())
    test_lengths = test_df['l'].values
    test_shop = env_matrix[test_df['event_id_2'].astype(int).values]
    test_label = test_df['label'].values
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(test_traj),
                                         torch.LongTensor(test_lengths),
                                         torch.FloatTensor(test_shop),
                                         torch.FloatTensor(test_label))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    traj_dim = train_traj.shape[2]
    shop_dim = train_shop.shape[1]

    dataloaders = (train_loader, val_loader, test_loader)
    return dataloaders, traj_dim, shop_dim, test_df[['event_id_1','event_id_2', 'label']]
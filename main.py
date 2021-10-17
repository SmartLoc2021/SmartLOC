'''
SmartLoc main.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pipline: Graph embedding(GIN) -> Localization (Transformer)
Owner: Dongzhe Jiang
Date: 2021-01-15
'''

import argparse
from log import *
from train import *
# from simulate import *
from models.localization import Localization
import pandas as pd

def main():
    parser = argparse.ArgumentParser('Interface for SmartLOC framework')

    # data path
    parser.add_argument('--data_path', type=str, default='Data/rider_trip_from_beacon_with_wifi_hqg_20201001_20201231.csv', help='dataset path') 
    parser.add_argument('--feature_path', type=str, default='Data/rendezvous_event_data_hqg_20201001_20201231.csv', help='feature path') 
    parser.add_argument('--inshop_path', type=str, default='Result/inshop_result_lgb.csv', help='inshop model result path') 
    # data preprocess param
    parser.add_argument('--THRE', type=float, default=360, help='time diff filter')
    parser.add_argument('--TIME_INTERVAL', type=float, default=5, help='time interval between two events')
    parser.add_argument('--test_date', type=int, default=20201213, help='test date')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='ratio of the test against whole')
    parser.add_argument('--data_usage', type=float, default=1.0, help='use partial dataset')
    # features
    parser.add_argument('--cat_num', type=int, default=7, help='cat_num')
    parser.add_argument('--loc_cat_num', type=int, default=7, help='loc_cat_num')
    # GNN param
    parser.add_argument('--model', type=str, default='GIN', help='model to use for merchant GNN', choices=['DE-GNN', 'GIN', 'GCN', 'GraphSAGE', 'GAT', 'PGNN'])
    parser.add_argument('--layers', type=int, default=3, help='largest number of layers')
    parser.add_argument('--hidden_features', type=int, default=128, help='hidden dimension')
    parser.add_argument('--feature', type=str, default='sp', help='distance encoding category: shortest path or random walk (landing probabilities)')  # sp (shortest path) or rw (random walk)
    parser.add_argument('--metric', type=str, default='mae', help='metric for evaluating performance', choices=['acc', 'auc'])
    parser.add_argument('--directed', type=bool, default=False, help='(Currently unavailable) whether to treat the graph as directed')
    parser.add_argument('--prop_depth', type=int, default=1, help='propagation depth (number of hops) for one layer')
    parser.add_argument('--use_degree', type=bool, default=True, help='whether to use node degree as the initial feature')
    parser.add_argument('--use_attributes', type=bool, default=True, help='whether to use node attributes as the initial feature')
    # Localization param
    parser.add_argument('--loc_layers', type=int, default=3, help='largest number of localization layers')
    parser.add_argument('--lstm_num_layers', type=int, default=3, help='largest number of lstm layers')
    parser.add_argument('--range', type=int, default=5, help='lstm range')
    parser.add_argument('--loc_hidden_features', type=int, default=128, help='loc_hidden_features')
    parser.add_argument('--rw_depth', type=int, default=3, help='random walk steps')  # for random walk feature
    parser.add_argument('--max_sp', type=int, default=3, help='maximum distance to be encoded for shortest path feature')
    parser.add_argument('--anchor_num', dest='anchor_num', default=32, type=int)
    # global parm
    parser.add_argument('--cat_embed_dim', dest='cat_embed_dim', default=3, type=int)
    parser.add_argument('--seed', type=int, default=0, help='seed to initialize all the random modules')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--parallel', default=False, action='store_true',
                        help='(Currently unavailable) whether to use multi cpu cores to prepare data')
    # Model training
    parser.add_argument('--epoch', type=int, default=900, help='number of epochs for GNN to train')
    parser.add_argument('--loc_epoch', type=int, default=1200, help='number of epochs for LOC to train')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--bs', type=int, default=64, help='minibatch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--loc_lr', type=float, default=5e-4, help='learning rate of localization')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer to use')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')

    # # simulation (valid only when dataset == 'simulation')
    # parser.add_argument('--k', type=int, default=3, help='node degree (k) or synthetic k-regular graph')
    # parser.add_argument('--n', nargs='*', help='a list of number of nodes in each connected k-regular subgraph')
    # parser.add_argument('--N', type=int, default=1000, help='total number of nodes in simultation')
    # parser.add_argument('--T', type=int, default=6, help='largest number of layers to be tested')
    # parser.add_argument('--approximate', dest='approximate', default=2, type=int,
    #                     help='k-hop shortest path distance. -1 means exact shortest path') # -1, 2

    # logging & debug
    parser.add_argument('--log_dir', type=str, default='log/', help='log directory')
    parser.add_argument('--model_dir', type=str, default='Model/', help='model directory')
    parser.add_argument('--result_dir', type=str, default='Result/', help='result directory')
    parser.add_argument('--cache_dir', type=str, default='Cache/', help='Cache directory')
    parser.add_argument('--version_name', type=str, default='adddist_3month_inshop_gnnhidden140_range5_transformer3_loclayer4', help='version name')
    parser.add_argument('--summary_file', type=str, default='result_summary.log', help='brief summary of training result')
    parser.add_argument('--debug', default=False, action='store_true',help='whether to use debug mode')
    parser.add_argument('--use_cache', type=bool, default=False, help='whether to use cache')


    sys_argv = sys.argv
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    check(args)
    logger = set_up_log(args, sys_argv)
    set_random_seed(args)
    G, data_df, ren_df, train_df, test_df, ren_catnums = read_file(args, logger)
    data = get_gnn_data(G, data_df, args=args, logger=logger)
    if args.model == 'PGNN':
        len_distmax = data.dists_argmax.shape[1]
    else:
        len_distmax = 0
    if args.use_attributes:
        cat_num = G.graph['cat_attributes'].max()+2
        cat_features = G.graph['cat_attributes'].shape[1]
    else:
        cat_num = 0
        cat_features = 0
    model_path = os.path.join(args.model_dir, "{}_{}.model".format(args.version_name,args.model))
    gnn_model = get_gnn_model(layers=args.layers, in_features=data.x.shape[-1], out_features=1,
                        prop_depth=args.prop_depth, args=args, logger=logger, len_distmax = len_distmax, cat_num = cat_num, cat_features = cat_features)
    if not (args.use_cache and os.path.exists(model_path)):
        results = train_gnn_model(gnn_model, data, args, logger)
        save_performance_result(args, logger, results)
    gnn_model.load_state_dict(torch.load(model_path))
    shop_matrix = gnn_result(gnn_model, data, args)
    dataloaders, traj_dim, shop_dim, test_df = get_lstm_data(ren_df, shop_matrix, test_df, args)
    lstm_input_dim = traj_dim-args.cat_num+args.cat_num*args.cat_embed_dim
    shop_input_dim = shop_dim-args.cat_num+args.cat_num*args.cat_embed_dim
    model = Localization(cat_num=args.cat_num, cattype_nums=ren_catnums, cat_embed_dim=args.cat_embed_dim, 
                    lstm_input_dim=lstm_input_dim, 
                         shop_input_dim=shop_input_dim, 
                         hidden_dim=args.loc_hidden_features, 
                         num_layers=args.loc_layers, 
                         output_dim=1, lstm_num_layers = args.lstm_num_layers, dropout=args.dropout)
    
    results, predictions = train_loc_model(model, dataloaders, args, logger)
    save_performance_result(args, logger, results, tag = 'LOC')
    
    # save result
    test_df['pred'] = predictions
    cache_path = os.path.join(args.cache_dir,"{}_{}_{}.csv".format('idmaps',args.version_name,args.model))
    with open(cache_path,'rb') as f:
        node_id_mapping, rider_id_mapping, event_id, le = pickle.load(f)
    test_df = pd.merge(left = test_df.rename(columns = {'event_id_1':'index'}), 
                    right = event_id, on = 'index').drop('index',axis = 1).rename(columns = {'event_id':'event_id_1'})
    test_df = pd.merge(left = test_df.rename(columns = {'event_id_2':'index'}), 
                    right = event_id, on = ['index','ds']).drop('index',axis = 1).rename(columns = {'event_id':'event_id_2'})
    result_path = os.path.join(args.result_dir, "{}_{}_{}_debug.csv".format('LOC',args.version_name,args.model))
    test_df.to_csv(result_path)
    print("Save result successfully!")

if __name__ == '__main__':
    main()

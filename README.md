# SmartLOC
This repository is the official PyTorch implementation of the *SmartLoc* reported in the paper: <br>
[*SmartLOC: Indoor Localization with Smartphone Anchors for On-Demand Delivery*](). 

## Installation
Requirements: Python >= 3.5, [Anaconda3](https://www.anaconda.com/)

- Update conda:
```bash
conda update -n base -c defaults conda
```

- Install basic dependencies to virtual environment and activate it: 
```bash
conda env create -f environment.yml
conda activate degnn-env
```

- Install PyTorch >= 1.4.0 and torch-geometric >= 1.5.0
```bash
conda install pytorch=1.4.0 torchvision cudatoolkit=10.1 -c pytorch
pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
pip install torch-geometric
```

The latest tested combination is: Python 3.8.2 + Pytorch 1.4.0 + torch-geometric 1.5.0.

## Quick Start
```
python main.py
```

## Usage Summary
```
Interface for SmartLOC framework [-h] [--data_path DATA_PATH] [--feature_path FEATURE_PATH] [--inshop_path INSHOP_PATH] [--THRE THRE] [--TIME_INTERVAL TIME_INTERVAL] [--test_date TEST_DATE] [--test_ratio TEST_RATIO] [--data_usage DATA_USAGE] [--cat_num CAT_NUM]
                                        [--loc_cat_num LOC_CAT_NUM] [--model {DE-GNN,GIN,GCN,GraphSAGE,GAT,PGNN}] [--layers LAYERS] [--hidden_features HIDDEN_FEATURES] [--feature FEATURE] [--metric {acc,auc}] [--directed DIRECTED] [--prop_depth PROP_DEPTH]
                                        [--use_degree USE_DEGREE] [--use_attributes USE_ATTRIBUTES] [--loc_layers LOC_LAYERS] [--lstm_num_layers LSTM_NUM_LAYERS] [--range RANGE] [--loc_hidden_features LOC_HIDDEN_FEATURES] [--rw_depth RW_DEPTH] [--max_sp MAX_SP]
                                        [--anchor_num ANCHOR_NUM] [--cat_embed_dim CAT_EMBED_DIM] [--seed SEED] [--gpu GPU] [--parallel] [--epoch EPOCH] [--loc_epoch LOC_EPOCH] [--batch_size BATCH_SIZE] [--bs BS] [--lr LR] [--loc_lr LOC_LR] [--optimizer OPTIMIZER]
                                        [--l2 L2] [--dropout DROPOUT] [--log_dir LOG_DIR] [--model_dir MODEL_DIR] [--result_dir RESULT_DIR] [--cache_dir CACHE_DIR] [--version_name VERSION_NAME] [--summary_file SUMMARY_FILE] [--debug] [--use_cache USE_CACHE]
```

## Optinal Arguments
```
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        dataset path
  --feature_path FEATURE_PATH
                        feature path
  --inshop_path INSHOP_PATH
                        inshop model result path
  --THRE THRE           time diff filter
  --TIME_INTERVAL TIME_INTERVAL
                        time interval between two events
  --test_date TEST_DATE
                        test date
  --test_ratio TEST_RATIO
                        ratio of the test against whole
  --data_usage DATA_USAGE
                        use partial dataset
  --cat_num CAT_NUM     cat_num
  --loc_cat_num LOC_CAT_NUM
                        loc_cat_num
  --model {DE-GNN,GIN,GCN,GraphSAGE,GAT,PGNN}
                        model to use for merchant GNN
  --layers LAYERS       largest number of layers
  --hidden_features HIDDEN_FEATURES
                        hidden dimension
  --feature FEATURE     distance encoding category: shortest path or random walk (landing probabilities)
  --metric {acc,auc}    metric for evaluating performance
  --directed DIRECTED   (Currently unavailable) whether to treat the graph as directed
  --prop_depth PROP_DEPTH
                        propagation depth (number of hops) for one layer
  --use_degree USE_DEGREE
                        whether to use node degree as the initial feature
  --use_attributes USE_ATTRIBUTES
                        whether to use node attributes as the initial feature
  --loc_layers LOC_LAYERS
                        largest number of localization layers
  --lstm_num_layers LSTM_NUM_LAYERS
                        largest number of lstm layers
  --range RANGE         lstm range
  --loc_hidden_features LOC_HIDDEN_FEATURES
                        loc_hidden_features
  --rw_depth RW_DEPTH   random walk steps
  --max_sp MAX_SP       maximum distance to be encoded for shortest path feature
  --anchor_num ANCHOR_NUM
  --cat_embed_dim CAT_EMBED_DIM
  --seed SEED           seed to initialize all the random modules
  --gpu GPU             gpu id
  --parallel            (Currently unavailable) whether to use multi cpu cores to prepare data
  --epoch EPOCH         number of epochs for GNN to train
  --loc_epoch LOC_EPOCH
                        number of epochs for LOC to train
  --batch_size BATCH_SIZE
                        batch size
  --bs BS               minibatch size
  --lr LR               learning rate
  --loc_lr LOC_LR       learning rate of localization
  --optimizer OPTIMIZER
                        optimizer to use
  --l2 L2               l2 regularization weight
  --dropout DROPOUT     dropout rate
  --log_dir LOG_DIR     log directory
  --model_dir MODEL_DIR
                        model directory
  --result_dir RESULT_DIR
                        result directory
  --cache_dir CACHE_DIR
                        Cache directory
  --version_name VERSION_NAME
                        version name
  --summary_file SUMMARY_FILE
                        brief summary of training result
  --debug               whether to use debug mode
  --use_cache USE_CACHE
                        whether to use cache
```
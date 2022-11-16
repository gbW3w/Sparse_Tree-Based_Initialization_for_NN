"""
All model evaluations should be saved in a .pkl file with the following items:
* nb_epochs: int, number of epochs
* metrics_train_: List[float|np.ndarray], array of training loss during training
* metrics_val_: List[float|np.ndarray], array of validation loss during training
* metrics_test_: List[float|np.ndarray], array of test loss during training
* metrics_name_: List[str], array of metric names
* metrics_add_: List[float|np.ndarray], array of an additional loss during training (optional)
"""

import pickle
from matplotlib import pyplot as plt
import yaml
from models.MLP import MLP
from models.saint import SAINT
from torch import nn
import torch
from utils.utils import DeviceCollector, get_metrics, verify_method_dataName
from models.tree_models import DeepForest, RandomForest, XGBoostForest
from utils.utils_data import DataLoader_RepeatedStratifiedKFold, handle_loggers
from utils.metrics import *

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# helper functions
def check_hp(hp, hp_names_verif, model_name):
    hp_missing = set(hp_names_verif).difference(set(hp.keys()))
    if len(hp_missing) > 0:
        raise ValueError(f'{model_name} requires hyper-parameter(s): {hp_missing}')

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def eval_tree(tree_model, data, metrics):

    metric_train = dict()
    metric_val = dict()
    metric_test = dict()
    for name, metric in metrics.items():
        # train
        pred = tree_model.predict(data['X_train'])
        metric_train[name] = metric(pred, data['y_train'])
    
        # validation
        pred = tree_model.predict(data['X_val'])
        metric_val[name] = metric(pred, data['y_val'])

        # test
        pred = tree_model.predict(data['X_test'])
        metric_test[name] = metric(pred, data['y_test'])

    return metric_train, metric_val, metric_test

# eval tree models
def eval_RF(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['max_depth', 'n_estim', 'max_features'], 'eval_RF')

    # get model
    model = RandomForest(DataLoader.task, DataLoader.n_classes, hp['max_depth'], hp['n_estim'], hp['max_features'])

    metrics_train_ = []
    metrics_val_ =  []
    metrics_test_ = []
    for data in DataLoader:
        # train RF
        model.fit(data['X_train'], data['y_train'])
        # eval RF
        metrics_train, metrics_val, metrics_test = eval_tree(model, data, metrics)

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_}

def eval_XGBoost(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['max_depth', 'n_estim', 'reg_l1', 'reg_l2', 'learning_rate'], 'eval_RF')

     # set device
    if torch.cuda.is_available():
        device = DeviceCollector.GPU
    else:
        device = DeviceCollector.CPU

    # get model
    model = XGBoostForest(DataLoader.task, DataLoader.n_classes, hp['max_depth'], hp['n_estim'], 
        hp['reg_l1'], hp['reg_l2'], hp['learning_rate'], device)

    metrics_train_ = []
    metrics_val_ =  []
    metrics_test_ = []
    for data in DataLoader:
        # train RF
        model.fit(data['X_train'], data['y_train'])
        # eval RF
        metrics_train, metrics_val, metrics_test = eval_tree(model, data, metrics)

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_}

def eval_DF(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['forest_depth', 'n_forests', 'n_estims', 'max_tree_depth', 'max_features'], 'eval_DF')

    # get model
    model = DeepForest(DataLoader.task, DataLoader.n_classes, hp['forest_depth'], hp['n_forests'], 
        hp['n_estims'], hp['max_tree_depth'], hp['max_features'], early_stopping_rounds=3)

    metrics_train_ = []
    metrics_val_ =  []
    metrics_test_ = []
    for data in DataLoader:
        # train RF
        model.fit(data['X_train'], data['y_train'])
        # eval RF
        metrics_train, metrics_val, metrics_test = eval_tree(model, data, metrics)

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_}

early_stop = 10
# eval MLP models
def eval_MLP_rand_init(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['learning_rate', 'depth', 'width', 'epochs', 'reg_l1', 'batch_size'], 'eval_MLP_Rand_Init')

    # create model
    model = MLP(
        in_features=DataLoader.in_features, 
        out_features=DataLoader.out_features, 
        hidden_features=[hp['width']]*(hp['depth']-1), 
        activation=nn.Tanh()
    )

    # train loop
    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    weight_hist_ = []
    wdiff_hist_ = []
    init_time_ = []
    for data in DataLoader:
        # initialize nn uniformily
        init_time = model.init_uniform()
        # train with random init
        metrics_train, metrics_val, metrics_test, weight_hist, wdiff_hist = model.fit(
            data.values(), 
            epochs=hp['epochs'], 
            task=DataLoader.task, 
            learning_rate=hp['learning_rate'],
            metrics=metrics,
            regularize_l1=hp['reg_l1'],
            batch_size=hp['batch_size'],
            early_stop=early_stop
        )

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)
        weight_hist_.append(weight_hist)
        wdiff_hist_.append(wdiff_hist)
        init_time_.append(init_time)

    print('number of MLP_rand_init parameters', count_parameters(model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_, 'weight_hist_': weight_hist_,
        'wdiff_hist_': wdiff_hist_, 'init_time_': init_time_}

def eval_MLP_RF_init(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['learning_rate', 'depth', 'width', 'epochs', 'reg_l1', 'RF_init_max_depth', 
        'RF_init_n_estim', 'RF_init_max_features', 'strength01', 'strength12', 'batch_size'], 'eval_MLP_RF_Init')

    # create MLP
    if 'width_init' in hp.keys():
        hidden_features = [hp['width_init']]*2 + [hp['width']]*(hp['depth']-3)
    else:
        hidden_features = [hp['width']]*(hp['depth']-1)
    model = MLP(
        in_features=DataLoader.in_features, 
        out_features=DataLoader.out_features, 
        hidden_features=hidden_features, 
        activation=nn.Tanh()
    )

    # get tree metics
    metrics_tree = get_metrics(DataLoader, 'RF')

    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    metrics_add_ = []
    weight_hist_ = []
    wdiff_hist_ = []
    init_time_ = []
    for data in DataLoader:
        # reset model
        model.init_uniform()
        
        # initialize MLP
        tree_init, init_time = model.init_RF(
            data.values(), 
            task=DataLoader.task, 
            n_classes=DataLoader.n_classes, 
            max_depth=hp['RF_init_max_depth'], 
            n_estim=hp['RF_init_n_estim'],
            max_features=hp['RF_init_max_features'],
            strength01=hp['strength01'], 
            strength12=hp['strength12'],
            drop_layers=hp['drop_layers']
        )

        # train MLP
        metrics_train, metrics_val, metrics_test, weight_hist, wdiff_hist = model.fit(
            data.values(),
            epochs=hp['epochs'], 
            task=DataLoader.task, 
            learning_rate=hp['learning_rate'], 
            metrics=metrics,
            regularize_l1=hp['reg_l1'],
            batch_size=hp['batch_size'],
            early_stop=early_stop
        )

        # evaluate the init RF
        metric_add = dict()
        for name, metric in metrics_tree.items():
            pred = tree_init.predict(data['X_test'])
            metric_add[name] = metric(pred, data['y_test'])

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)
        metrics_add_.append(metric_add)
        weight_hist_.append(weight_hist)
        wdiff_hist_.append(wdiff_hist)
        init_time_.append(init_time)

    print('number of MLP_RF_init parameters', count_parameters(model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_, 'metrics_add_': metrics_add_,
        'weight_hist_': weight_hist_, 'wdiff_hist_': wdiff_hist_,
        'init_time_': init_time_}

def eval_MLP_XGB_init(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['learning_rate', 'depth', 'width', 'epochs', 'reg_l1', 'XGB_init_max_depth', 
        'XGB_init_n_estim', 'strength01', 'strength12', 'XGB_init_reg_l1', 
        'XGB_init_reg_l2', 'XGB_init_learning_rate', 'batch_size'], 'eval_MLP_XGB_Init')

    # create MLP
    if 'width_init' in hp.keys():
        hidden_features = [hp['width_init']]*2 + [hp['width']]*(hp['depth']-3)
    else:
        hidden_features = [hp['width']]*(hp['depth']-1)
    model = MLP(
        in_features=DataLoader.in_features, 
        out_features=DataLoader.out_features, 
        hidden_features=hidden_features, 
        activation=nn.Tanh()
    )

    # get tree metics
    metrics_tree = get_metrics(DataLoader, 'XGB')

    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    metrics_add_ = []
    weight_hist_ = []
    wdiff_hist_ = []
    init_time_ = []
    for data in DataLoader:
        # reset model
        model.init_uniform()
        
        # initialize MLP
        tree_init, init_time = model.init_XGBoost(
            data.values(), 
            task=DataLoader.task, 
            n_classes=DataLoader.n_classes, 
            max_depth=hp['XGB_init_max_depth'], 
            n_estim=hp['XGB_init_n_estim'],
            strength01=hp['strength01'], 
            strength12=hp['strength12'],
            reg_l1=hp['XGB_init_reg_l1'], 
            reg_l2=hp['XGB_init_reg_l2'], 
            learning_rate=hp['XGB_init_learning_rate'], 
            use_gpu=torch.cuda.is_available(),
            drop_layers=hp['drop_layers']
        )

        # train MLP
        metrics_train, metrics_val, metrics_test, weight_hist, wdiff_hist = model.fit(
            data.values(),
            epochs=hp['epochs'], 
            task=DataLoader.task, 
            learning_rate=hp['learning_rate'], 
            metrics=metrics,
            regularize_l1=hp['reg_l1'],
            batch_size=hp['batch_size'],
            early_stop=early_stop
        )

        # evaluate the init RF
        metric_add = dict()
        for name, metric in metrics_tree.items():
            pred = tree_init.predict(data['X_test'])
            metric_add[name] = metric(pred, data['y_test'])

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)
        metrics_add_.append(metric_add)
        weight_hist_.append(weight_hist)
        wdiff_hist_.append(wdiff_hist)
        init_time_.append(init_time)

    print('number of MLP_XGB_init parameters', count_parameters(model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_, 'metrics_add_': metrics_add_,
        'weight_hist_': weight_hist_, 'wdiff_hist_': wdiff_hist_,
        'init_time_': init_time_}

def eval_MLP_DF_init(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['learning_rate', 'depth', 'width', 'epochs', 'reg_l1', 'DF_init_forest_depth', 
        'DF_init_n_forests', 'DF_init_n_estims', 'DF_init_max_tree_depth', 'DF_init_max_features', 
        'batch_size', 'strength01', 'strength12', 'strength23', 'strength_id', 'drop_layers'], 'eval_MLP_DF_Init')

    # create MLP
    print(hp['depth'])
    model = MLP(
        in_features=DataLoader.in_features, 
        out_features=DataLoader.out_features, 
        hidden_features=[hp['width']]*(hp['depth']-1), 
        activation=nn.Tanh()
    )

    # get tree metics
    metrics_tree = get_metrics(DataLoader, 'DF')

    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    metrics_add_ = []
    weight_hist_ = []
    wdiff_hist_ = []
    init_time_ = []
    for data in DataLoader:
        # reset model
        model.init_uniform()
        
        # initialize MLP
        tree_init, init_time = model.init_DF(
            data.values(), 
            task=DataLoader.task, 
            n_classes=DataLoader.n_classes, 
            forest_depth=hp['DF_init_forest_depth'],
            n_forests=hp['DF_init_n_forests'],
            n_estim=hp['DF_init_n_estims'], 
            max_depth=hp['DF_init_max_tree_depth'], 
            strength01=hp['strength01'],
            strength12=hp['strength12'],
            strength23=hp['strength23'],
            strength_id=hp['strength_id'],
            drop_layers=hp['drop_layers'],
            tree_max_features=hp['DF_init_max_features']
        )

        # train MLP
        metrics_train, metrics_val, metrics_test, weight_hist, wdiff_hist = model.fit(
            data.values(),
            epochs=1,#hp['epochs'], 
            task=DataLoader.task, 
            learning_rate=hp['learning_rate'], 
            metrics=metrics,
            regularize_l1=hp['reg_l1'],
            batch_size=hp['batch_size'],
            early_stop=early_stop
        )

        # evaluate the init RF
        metric_add = dict()
        for name, metric in metrics_tree.items():
            pred = tree_init.predict(data['X_test'])
            metric_add[name] = metric(pred, data['y_test'])

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)
        metrics_add_.append(metric_add)
        weight_hist_.append(weight_hist)
        wdiff_hist_.append(wdiff_hist)
        init_time_.append(init_time)

    print('number of MLP_DF_init parameters', count_parameters(model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_, 'metrics_add_': metrics_add_,
        'weight_hist_': weight_hist_, 'wdiff_hist_': wdiff_hist_,
        'init_time_': init_time_}

def eval_MLP_WT_init(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['learning_rate', 'depth', 'width', 'epochs', 'reg_l1', 'batch_size'], 'eval_MLP_Rand_Init')

    # create model
    model = MLP(
        in_features=DataLoader.in_features, 
        out_features=DataLoader.out_features, 
        hidden_features=[hp['width']]*(hp['depth']-1), 
        activation=nn.Tanh()
    )

    # train loop
    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    weight_hist_ = []
    wdiff_hist_ = []
    init_time_ = []
    for data in DataLoader:
        # initialize nn with winning ticket
        init_time = model.init_winning_ticket(
            data.values(), 
            task=DataLoader.task, 
            epochs=hp['epochs'], 
            batch_size=hp['batch_size'], 
            learning_rate=hp['lr_pruning'], 
            metrics=metrics,
            early_stop=early_stop,
            pruning_rounds=hp['pruning_rounds'], 
            pruning_rate=hp['pruning_rate']
        )
        init_time_.append(init_time)
        # train
        metrics_train, metrics_val, metrics_test, weight_hist, wdiff_hist = model.fit(
            data.values(), 
            epochs=hp['epochs'], 
            task=DataLoader.task, 
            learning_rate=hp['learning_rate'],
            metrics=metrics,
            regularize_l1=hp['reg_l1'],
            batch_size=hp['batch_size'],
            early_stop=early_stop
        )

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)
        weight_hist_.append(weight_hist)
        wdiff_hist_.append(wdiff_hist)

    print('number of MLP_WT_init parameters', count_parameters(model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_, 'weight_hist_': weight_hist_,
        'wdiff_hist_': wdiff_hist_, 'init_time_': init_time_}  

def eval_MLP_LSUV_init(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['learning_rate', 'depth', 'width', 'epochs', 'reg_l1', 'batch_size'], 'eval_MLP_Rand_Init')

    # create model
    model = MLP(
        in_features=DataLoader.in_features, 
        out_features=DataLoader.out_features, 
        hidden_features=[hp['width']]*(hp['depth']-1), 
        activation=nn.Tanh()
    )

    # train loop
    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    weight_hist_ = []
    wdiff_hist_ = []
    init_time_ = []
    for data in DataLoader:
        # initialize nn with winning ticket
        init_time = model.init_lsuv(data['X_train'])
        init_time_.append(init_time)
        # train
        metrics_train, metrics_val, metrics_test, weight_hist, wdiff_hist = model.fit(
            data.values(), 
            epochs=hp['epochs'], 
            task=DataLoader.task, 
            learning_rate=hp['learning_rate'],
            metrics=metrics,
            regularize_l1=hp['reg_l1'],
            batch_size=hp['batch_size'],
            early_stop=early_stop
        )

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)
        weight_hist_.append(weight_hist)
        wdiff_hist_.append(wdiff_hist)

    print('number of MLP_LSUV_init parameters', count_parameters(model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_, 'weight_hist_': weight_hist_,
        'wdiff_hist_': wdiff_hist_, 'init_time_': init_time_} 

def eval_MLP_xavier_init(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['learning_rate', 'depth', 'width', 'epochs', 'reg_l1', 'batch_size'], 'eval_MLP_Rand_Init')

    # create model
    model = MLP(
        in_features=DataLoader.in_features, 
        out_features=DataLoader.out_features, 
        hidden_features=[hp['width']]*(hp['depth']-1), 
        activation=nn.Tanh()
    )

    # train loop
    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    weight_hist_ = []
    wdiff_hist_ = []
    init_time_ = []
    for data in DataLoader:
        # initialize nn with winning ticket
        init_time = model.init_xavier()
        init_time_.append(init_time)
        # train
        metrics_train, metrics_val, metrics_test, weight_hist, wdiff_hist = model.fit(
            data.values(), 
            epochs=hp['epochs'], 
            task=DataLoader.task, 
            learning_rate=hp['learning_rate'],
            metrics=metrics,
            regularize_l1=hp['reg_l1'],
            batch_size=hp['batch_size'],
            early_stop=early_stop
        )

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)
        weight_hist_.append(weight_hist)
        wdiff_hist_.append(wdiff_hist)

    print('number of MLP_LSUV_init parameters', count_parameters(model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_, 'weight_hist_': weight_hist_,
        'wdiff_hist_': wdiff_hist_, 'init_time_': init_time_} 

# eval SAINT
def eval_SAINT(DataLoader, metrics, hp):
    # check HP
    check_hp(hp, ['dim', 'depth', 'heads', 'dropout'], 'eval_SAINT')

    # create model
    if DataLoader.task == 'classification':
        cat_idx = np.where(DataLoader.is_categorical)[0]
        cat_dims = [np.unique(col).size for col in DataLoader.X[:,cat_idx].T]
    else:
        cat_idx, cat_dims = None, None

    # train loop
    metrics_train_ = []
    metrics_val_ = []
    metrics_test_ = []
    for data in DataLoader:
        model = SAINT(DataLoader.in_features, DataLoader.out_features, hp['dim'], hp['depth'], hp['heads'], hp['dropout'], cat_idx, cat_dims, DataLoader.task)
        # train with random init
        metrics_train, metrics_val, metrics_test = model.fit(
            data.values(), 
            epochs=hp['epochs'], 
            task=DataLoader.task, 
            metrics=metrics,
            batch_size=hp['batch_size'],
            early_stop=10
        )

        metrics_train_.append(metrics_train)
        metrics_val_.append(metrics_val)
        metrics_test_.append(metrics_test)

    print('number of SAINT parameters', count_parameters(model.model))

    return {'metrics_train_': metrics_train_, 'metrics_val_': metrics_val_, 
        'metrics_test_': metrics_test_}

method_fun_mapper = {
    'RF': eval_RF,
    'XGB': eval_XGBoost,
    'DF': eval_DF,
    'MLP_rand_init': eval_MLP_rand_init,
    'MLP_RF_init': eval_MLP_RF_init,
    'MLP_XGB_init': eval_MLP_XGB_init,
    'MLP_DF_init': eval_MLP_DF_init,
    'MLP_WT_init': eval_MLP_WT_init,
    'MLP_LSUV_init': eval_MLP_LSUV_init,
    'MLP_xavier_init': eval_MLP_xavier_init,
    'SAINT': eval_SAINT
}

# wrap-up function
def create_eval_pkl(method:str, data_name: str, n_splits: int, n_repeats: int, n_max: int, one_hot: bool=False, in_suffix: str='', out_suffix: str='',
                    eval_suffix: str=''):
    # verify method and data names
    verify_method_dataName(method, data_name)

    # get hyper parameters
    with open(f'model_HP{eval_suffix}/{method}/{data_name}{in_suffix}.yaml') as file:
        hp = yaml.load(file, Loader=yaml.FullLoader)
    # get evaluation function
    eval_fun = method_fun_mapper[method]
    # get data loader
    DataLoader =  DataLoader_RepeatedStratifiedKFold(data_name, n_splits, n_repeats, one_hot, n_max) 
    # set class weights for classification problems
    #if DataLoader.task == 'classification':
    #    DataLoader.set_class_weights(hp['beta'])
    metrics = get_metrics(DataLoader, method)
    # get metrics
    metrics_dict = eval_fun(DataLoader, metrics, hp)
    print(f'data set chars\n\tdims {DataLoader.X.shape}\n\tnb cat feature {sum(DataLoader.is_categorical)}\n\tnb classes {len(np.unique(DataLoader.y))}')
    metrics_dict['metrics_names'] = list(metrics.keys())
    metrics_dict['hp'] = hp
    # save metrics
    with open(f'eval{eval_suffix}/eval_{data_name}_{method}{in_suffix}{out_suffix}.pkl', 'wb') as file:
        pickle.dump(metrics_dict, file)

if __name__ == '__main__':
    handle_loggers(terminal=True)
    create_eval_pkl(method='MLP_xavier_init', data_name='housing', n_splits=5, n_repeats=5, n_max=None, one_hot=False, 
        in_suffix='', out_suffix='', eval_suffix='')
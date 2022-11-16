import datetime
import logging
import yaml
import numpy as np
from utils.utils import *
from utils.utils_data import DataLoader_RepeatedStratifiedKFold, handle_loggers
from eval_models import *
from search_spaces import *
import optuna
from optuna.trial import TrialState
import os.path


method_fun_mapper = {
    'RF': (search_space_RF, eval_RF),
    'RF_init': (search_space_RF_init, eval_RF),
    'XGB': (search_space_XGB, eval_XGBoost),
    'XGB_init': (search_space_XGB_init, eval_XGBoost),
    'DF': (search_space_DF, eval_DF),
    'DF_init': (search_space_DF_init, eval_DF),
    'MLP_rand_init': (search_space_MLP_rand_init, eval_MLP_rand_init),
    'MLP_RF_init': (search_space_MLP_RF_init, eval_MLP_RF_init),
    'MLP_XGB_init': (search_space_MLP_XGB_init, eval_MLP_XGB_init),
    'MLP_DF_init': (search_space_MLP_DF_init, eval_MLP_DF_init),
    'MLP_WT_init': (search_space_MLP_WT_init, eval_MLP_WT_init),
    'MLP_LSUV_init': (search_space_MLP_LSUV_init, eval_MLP_LSUV_init),
    'MLP_xavier_init': (search_space_MLP_xavier_init, eval_MLP_xavier_init),
    'SAINT': (search_space_SAINT, eval_SAINT)
}

load_fixed_hp = {
    'MLP_RF_init': 'RF_init',
    'MLP_XGB_init': 'XGB_init',
    'MLP_DF_init': 'DF_init'
}

early_stopping = True

def objective(trial: optuna.trial, DataLoader: DataLoader_RepeatedStratifiedKFold, method: str, hp_fix: dict, choose_loss_fun, loss_name:str):

    # get search space and eval functions
    search_space_fun, eval_fun = method_fun_mapper[method]

    # define hyper params
    hp = {**hp_fix, **search_space_fun(trial, hp_fix, DataLoader.n_classes, DataLoader.task, DataLoader.in_features)}
    # add fixed HPs to trial
    fixed_hps = set(hp.keys()).difference(set(trial.params.keys()))
    for hp_name in fixed_hps: 
        trial.set_user_attr(hp_name, hp[hp_name])

    # set class weights for classification problems
    if DataLoader.task == 'classification' and method_isMLP(method):
        DataLoader.set_class_weights(hp['beta'])

    # set up metrics
    metrics = get_metrics(DataLoader, method)

    # evaluate model
    if loss_name is None:
        loss_name = list(metrics.keys())[0]
    metrics_dict = eval_fun(DataLoader, metrics, hp)
    choose_loss_fun = choose_loss_fun if early_stopping else lambda x: x[-1]
    objective = np.mean([choose_loss_fun(metrics_val[loss_name]) for metrics_val in metrics_dict['metrics_val_']])

    return objective

def optimize_HP(method: str, data_name: str, database_rep: str, *, n_trials:int, n_splits: int, 
    n_repeat: int, n_max: int=None, loss_name: str=None, choose_loss_fun=np.min, suffix: str='', direction='minimize', one_hot=False):

    logger = logging.getLogger('optimize_HP')
    # verify mehtod and data names
    verify_method_dataName(method, data_name)

    # create/load study
    study_name = f'{data_name}_{method}{suffix}'
    file = database_rep + study_name + '.db'
    storage = 'sqlite:///' + file
    if os.path.isfile(file): 
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        print('trying to create study')
        study = optuna.create_study(study_name=study_name, storage=storage, direction=direction)

    # get datalaoder
    DataLoader = DataLoader_RepeatedStratifiedKFold(data_name, n_splits=n_splits, 
        n_repeats=n_repeat, n_max=n_max, one_hot=one_hot)
    n = DataLoader.get_n_splits()

    # get fixed HP
    if method in load_fixed_hp.keys():
        file_name = f'model_HP/{load_fixed_hp[method]}/{data_name}{suffix}.yaml'
        if not os.path.isfile(file_name):
            raise RuntimeError(f'File {file_name} not found.') 
        with open(file_name) as file:
            hp_fix = yaml.load(file, Loader=yaml.FullLoader)
        hp_fix = {load_fixed_hp[method]+'_'+key: item for key, item in hp_fix.items()}
    else:
        hp_fix = dict()

    # get objective function
    objective_fun_aug = lambda trial: objective(trial, DataLoader, method, hp_fix, choose_loss_fun, loss_name)

    # carry out HP search
    study.optimize(objective_fun_aug, n_trials=n_trials, timeout=100000)

    # print study results
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])    
    logger.info("Study statistics: ")
    logger.info(f"  Number of finished trials: {len(study.trials)}")
    logger.info(f"  Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"  Number of complete trials: {len(complete_trials)}")
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Value: {best_trial.value}")
    logger.info("  Params: ")
    for key, value in best_trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # save study results
    file_dump = yaml.dump({**best_trial.params, **best_trial.user_attrs})
    file_dump = '# HP search done on ' + datetime.datetime.now().__str__() + '\n' + file_dump
    filename = f'model_HP/{method}/{data_name}{suffix}.yaml'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(file_dump)

if __name__ == '__main__':
    handle_loggers(None, True)
    database_rep = './database/'
    optimize_HP('MLP_RF_init', 'housing', database_rep, n_trials=2, n_splits=5, n_repeat=1, 
        loss_name='MSELoss', suffix='', n_max=1, direction='minimize', choose_loss_fun=np.min, one_hot=False)#_fix_lstrength
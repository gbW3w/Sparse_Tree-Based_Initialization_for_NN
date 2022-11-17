import math

MLP_WIDTH = 2048
MLP_DEPTH = 10
NB_EPOCHS = 100
NB_ESTIM = 1000
BATCH_SIZE = 256
DROP_LAYERS = 1

def search_space_RF(trial, hp_fix, n_classes, task, in_features):
    return {
            'max_depth': trial.suggest_int('max_depth', 1, 12),
            'n_estim': NB_ESTIM, #fixed NB_ESTIM
            'max_features': trial.suggest_float('max_features', 0., 1.)
        }

def search_space_RF_init(trial, hp_fix, n_classes, task, in_features):
    max_depth = trial.suggest_int('max_depth', 1, int(math.log2(MLP_WIDTH)))
    return {
            'max_depth': max_depth,
            'n_estim': int(MLP_WIDTH/2**max_depth), # fix largeur/2**max_depth
            'max_features': trial.suggest_float('max_features', 0., 1.)# free max features between 0.5*sqrt(dim) and 1.5*sqrt(dim)
        }

def search_space_XGB(trial, hp_fix, n_classes, task, in_features):
    return {
        'max_depth': trial.suggest_int('max_depth', 1, 12),
        'n_estim': NB_ESTIM, #fix NB_ESTIM
        'reg_l1': trial.suggest_float('reg_l1', 1e-8, 1.0, log=True), #remove one regularization??
        'reg_l2': trial.suggest_float('reg_l2', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }

def search_space_XGB_init(trial, hp_fix, n_classes, task, in_features):
    max_depth = trial.suggest_int('max_depth', 1, int(math.log2(MLP_WIDTH/n_classes)) if n_classes > 2 else int(math.log2(MLP_WIDTH)))
    print('max depth', 1, int(math.log2(MLP_WIDTH/n_classes)) if n_classes > 2 else int(math.log2(MLP_WIDTH)), 'choice', max_depth)
    print('n_estim', int(MLP_WIDTH/(n_classes*2**max_depth)) if n_classes > 2 else int(MLP_WIDTH/(2**max_depth)))
    return {
        'max_depth': max_depth,
        'n_estim': int(MLP_WIDTH/(n_classes*2**max_depth)) if n_classes > 2 else int(MLP_WIDTH/(2**max_depth)),
        'reg_l1': trial.suggest_float('reg_l1', 1e-8, 1.0, log=True), #remove one regularization
        'reg_l2': trial.suggest_float('reg_l2', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }

def search_space_DF(trial, hp_fix, n_classes, task, in_features):
    return {
        'forest_depth': trial.suggest_int('forest_depth', 1, 10), 
        'n_forests': 1,#trial.suggest_int('n_forests', 1, 4),
        'n_estims': NB_ESTIM, 
        'max_tree_depth': trial.suggest_int('max_tree_depth', 1, 12),
        'max_features': trial.suggest_float('max_features', 0., 1.)
    }

def search_space_DF_init(trial, hp_fix, n_classes, task, in_features):
    max_tree_depth = trial.suggest_int('max_tree_depth', 1, int(math.log2(MLP_WIDTH)))
    return {
        'forest_depth': trial.suggest_int('forest_depth', 1, 3), 
        'n_forests': 1, # test 1, 2, 4, half completely random
        'n_estims': int(MLP_WIDTH/2**max_tree_depth), 
        'max_tree_depth': max_tree_depth,
        'max_features': trial.suggest_float('max_features', 0., 1.)
    }

def search_space_MLP_rand_init(trial, hp_fix, n_classes, task, in_features):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True), 
        'depth': trial.suggest_int('depth', 1, MLP_DEPTH), 
        'width': trial.suggest_int('width', 1, MLP_WIDTH), # fixe
        'epochs': NB_EPOCHS, # fixe
        'reg_l1': 0, # fixe a 0
        'batch_size': BATCH_SIZE, # fixe
        'beta': 0.5#None if task == 'regression' else 1-trial.suggest_float('beta', 1e-8, 1e-3, log=True)
    }

def search_space_MLP_RF_init(trial, hp_fix, n_classes, task, in_features):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True), 
        'depth': trial.suggest_int('depth', 3, MLP_DEPTH), 
        #'width_init': MLP_WIDTH,#trial.suggest_int('width', 500, 2000),
        'width': MLP_WIDTH,
        'epochs': NB_EPOCHS, 
        'reg_l1': 0, 
        'batch_size': BATCH_SIZE,
        'strength01': 1e10,#trial.suggest_float('strength01', 1e0, 1e4, log=True),
        'strength12': 1e10,#trial.suggest_float('strength12', 1e-2, 1e2, log=True),
        'drop_layers': DROP_LAYERS,
        'beta': 0.5
    }

def search_space_MLP_XGB_init(trial, hp_fix, n_classes, task, in_features):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True), 
        'depth': trial.suggest_int('depth', 3, MLP_DEPTH), 
        #'width_init': MLP_WIDTH,#trial.suggest_int('width', 700, 2000), 
        'width': MLP_WIDTH,#trial.suggest_int('width', 1, MLP_WIDTH),
        'epochs': NB_EPOCHS, 
        'reg_l1': 0,#trial.suggest_float('reg_l1', 1e-8, 1e0, log=True), 
        'batch_size': BATCH_SIZE,
        'strength01': trial.suggest_float('strength01', 1e0, 1e4),
        'strength12': trial.suggest_float('strength12', 1e-2, 1e2),
        'drop_layers': DROP_LAYERS,
        'beta': 0.5,#None if task == 'regression' else 1-trial.suggest_float('beta', 1e-8, 1e-3, log=True)
        'dropout': None,#trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    }

def search_space_MLP_DF_init(trial, hp_fix, n_classes, task, in_features):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True), 
        'depth': trial.suggest_int('depth', hp_fix['DF_init_forest_depth']*3, hp_fix['DF_init_forest_depth']*3+7), 
        'width': MLP_WIDTH+in_features,#trial.suggest_int('width', 500, 2000), 
        'epochs': NB_EPOCHS, 
        'reg_l1': 0, 
        'batch_size': BATCH_SIZE,
        'strength01': trial.suggest_float('strength01', 1e-1, 1e3, log=True), 
        'strength12': trial.suggest_float('strength12', 1e-2, 1e2, log=True),
        'strength23': trial.suggest_float('strength23', 1e-2, 1e2, log=True), 
        'strength_id': trial.suggest_float('strength_id', 1e-2, 1e2, log=True), 
        'drop_layers': DROP_LAYERS,
        'beta': 0.5
    }

def search_space_MLP_WT_init(trial, hp_fix, n_classes, task, in_features):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True), 
        'depth': trial.suggest_int('depth', 1, MLP_DEPTH), 
        'width': trial.suggest_int('width', 1, MLP_WIDTH), # fixe
        'epochs': NB_EPOCHS, # fixe
        'reg_l1': 0, # fixe a 0
        'batch_size': BATCH_SIZE, # fixe
        'beta': 0.5,#None if task == 'regression' else 1-trial.suggest_float('beta', 1e-8, 1e-3, log=True),
        'lr_pruning': trial.suggest_float('lr_pruning', 1e-6, 1e-1, log=True), 
        'pruning_rounds': trial.suggest_categorical('pruning_rounds', [1,2,3,4,5,6,7,8,9,10,11,12,14,16,20,25]),
        'pruning_rate': trial.suggest_float('pruning_rate', 0, 1)
    }

def search_space_MLP_LSUV_init(trial, hp_fix, n_classes, task, in_features):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True), 
        'depth': trial.suggest_int('depth', 1, MLP_DEPTH), 
        'width': trial.suggest_int('width', 1, MLP_WIDTH), # fixe
        'epochs': NB_EPOCHS, # fixe
        'reg_l1': 0, # fixe a 0
        'batch_size': BATCH_SIZE, # fixe
        'beta': 0.5#None if task == 'regression' else 1-trial.suggest_float('beta', 1e-8, 1e-3, log=True)
    }

def search_space_MLP_xavier_init(trial, hp_fix, n_classes, task, in_features):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-1, log=True), 
        'depth': trial.suggest_int('depth', 1, MLP_DEPTH), 
        'width': trial.suggest_int('width', 1, MLP_WIDTH), # fixe
        'epochs': NB_EPOCHS, # fixe
        'reg_l1': 0, # fixe a 0
        'batch_size': BATCH_SIZE, # fixe
        'beta': 0.5#None if task == 'regression' else 1-trial.suggest_float('beta', 1e-8, 1e-3, log=True)
    }

def search_space_SAINT(trial, hp_fix, n_classes, task, in_features):
    return {
        'epochs': 100,
        'batch_size': 64 if in_features > 20 else 256,
        "dim": trial.suggest_categorical("dim", [32, 64, 128] if in_features < 20 else [8, 16]), #if in_features < 10 else [32, 64, 128]
        "depth": trial.suggest_categorical("depth", [1, 2, 3]),
        "heads": trial.suggest_categorical("heads", [2, 4, 8]),
        "dropout": trial.suggest_categorical("dropout", [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        'beta': 0.5
    }
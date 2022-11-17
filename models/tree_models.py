from sklearn import ensemble, tree
from utils.utils import *
import xgboost as xgb
import numpy as np
import pandas as pd
import models.forestlayer_master.forestlayer.layers.layer as fl_layers
import models.forestlayer_master.forestlayer.layers.graph as fl_graph
import models.forestlayer_master.forestlayer.estimators.estimator_configs as fl_estimators
from typing import List, Union

class XGBoostForest():

    def __init__(self, task: str, n_classes: int, max_depth: int, n_estim: int, reg_weight_l1: float, 
                reg_weight_l2: float, learning_rate: float, device: Device):
        
        self.n_classes = n_classes
        self.n_estim = n_estim
        self.learning_rate = learning_rate
        self.device = device

        kwargs_model = {
            'max_depth': max_depth,
            'reg_alpha': reg_weight_l1,
            'reg_lambda': reg_weight_l2,
            'learning_rate': learning_rate,
            'n_estimators': n_estim
        }
        self.task = get_task(task, n_classes)

        if device == DeviceCollector.GPU:
            kwargs_model['tree_method'] = 'gpu_hist'

        if self.task == TaskCollector.REGRESSION:
            kwargs_model['objective'] = 'reg:squarederror'
            self.model = xgb.XGBRegressor(**kwargs_model)
        elif self.task == TaskCollector.CLASSIF_BINARY:
            kwargs_model['objective'] = 'binary:logistic'
            #kwargs_model['num_class'] = n_classes
            self.model = xgb.XGBClassifier(**kwargs_model)
        elif self.task == TaskCollector.CLASSIF_MULTICLASS:
            kwargs_model['objective'] = 'multi:softprob'
            kwargs_model['num_class'] = n_classes
            self.model = xgb.XGBClassifier(**kwargs_model)
        else:
            raise ValueError()

    def fit(self, X: np.ndarray, y: np.ndarray):
        # fit model
        self.model.fit(X, y)

    def predict(self, X):
        if self.task == TaskCollector.REGRESSION:
            return self.model.predict(X)
        if self.task in (TaskCollector.CLASSIF_BINARY, TaskCollector.CLASSIF_MULTICLASS):
            return self.model.predict_proba(X)

    def get_forest_structure(self):
        trees_xgb = self.model.get_booster().trees_to_dataframe()

        # tree
        trees = pd.DataFrame({'tree': trees_xgb['Tree']})  

        # id  
        xgb_id_mapping = {item:i for i, item in enumerate(trees_xgb['ID'].unique())}
        trees['id'] = trees_xgb['ID'].apply(lambda x: xgb_id_mapping[x])   

        # features
        def get_features_from_string(string):
            if string == 'Leaf':
                return np.NaN
            # extract int from string
            return int(''.join(filter(str.isdigit, string)))
        trees['feature'] = trees_xgb['Feature'].apply(get_features_from_string).astype('Int32')

        # threshold
        trees['threshold'] = trees_xgb['Split']

        # is_leaf
        trees['is_leaf'] = trees['feature'].isnull()

        # childs
        trees['left_child'] = trees_xgb['Yes'].apply(lambda x: int(xgb_id_mapping[x]) if pd.notnull(x) else pd.NA).astype('Int32')
        trees['right_child'] = trees_xgb['No'].apply(lambda x: int(xgb_id_mapping[x]) if pd.notnull(x) else pd.NA).astype('Int32')

        # leaf value
        trees['leaf_value'] = trees_xgb['Gain'].mask(~trees['is_leaf'])

        # predicted_class
        if self.task in (TaskCollector.REGRESSION, TaskCollector.CLASSIF_BINARY):
            trees['pred_outFeature'] = 0
        else:
            trees['pred_outFeature'] = trees['tree'] % self.n_classes

        return trees

    def get_base_score(self):
        if self.task == TaskCollector.CLASSIF_BINARY:
            return 0
        else:
            return 0.5

    def get_n_forests(self):
        return 1
        
def get_sklearn_DT_structure(tree_list: List[Union[tree.DecisionTreeClassifier, tree.DecisionTreeRegressor]], task: Task,
                            start_node_count: int=0) -> pd.DataFrame:
    forest_specs = pd.DataFrame()
    n_estim = len(tree_list)
    total_node_count = start_node_count
    for tree_id, descision_tree in enumerate(tree_list):
        # get descision tree
        tree = descision_tree.tree_

        tree_specs = pd.DataFrame()
        # get features
        tree_specs['feature'] = tree.feature
        tree_specs['feature'] = tree_specs['feature'].mask(tree_specs['feature'] == -2).astype('Int32')

        # tree ids
        tree_specs['tree'] = tree_id
        # node ids
        tree_specs['id'] = np.arange(tree.node_count) + total_node_count
        #thresholds
        tree_specs['threshold'] = tree.threshold
        tree_specs['threshold'] = tree_specs['threshold'].mask(tree_specs['threshold'] == -2)
        # is_leaf
        tree_specs['is_leaf'] = tree.feature == -2
        # left_child
        tree_specs['left_child'] = tree.children_left
        tree_specs['left_child'] = tree_specs['left_child'].mask(tree_specs['left_child'] == -1).astype('Int32')
        tree_specs['left_child'] = tree_specs['left_child'] + total_node_count
        # right_child
        tree_specs['right_child'] = tree.children_right
        tree_specs['right_child'] = tree_specs['right_child'].mask(tree_specs['right_child'] == -1).astype('Int32')
        tree_specs['right_child'] = tree_specs['right_child'] + total_node_count
        # leaf_value
        if task == TaskCollector.REGRESSION:
            tree_specs['leaf_value'] = tree.value.squeeze()/n_estim
        #elif task == TaskCollector.CLASSIF_BINARY:
        #    tree_specs['leaf_value'] = tree.value.squeeze()[:,1]/(n_estim*np.sum(tree.value.squeeze(), axis=1))
        else:
            tree_specs['leaf_value'] = (tree.value.squeeze()/(n_estim*np.sum(tree.value.squeeze(), axis=1, keepdims=True))).tolist()
        tree_specs['leaf_value'] = tree_specs['leaf_value'].mask(~tree_specs['is_leaf'])

        # prediction out feature
        tree_specs['pred_outFeature'] = 0

        # keep track of total node count
        total_node_count += tree.node_count           

        # append to tree spect to forest specs
        if len(forest_specs) == 0:
            forest_specs = tree_specs
        else:
            forest_specs = pd.concat([forest_specs,tree_specs])

    return forest_specs

class RandomForest():

    def __init__(self, task: str, n_classes:int, max_depth: int, n_estim: int, max_features: int):
        kwargs_model = {
            'max_depth': max_depth,
            'n_estimators': n_estim,
            'max_features': max_features,
            'n_jobs': -1
        }
        self.task = get_task(task, n_classes)

        if self.task == TaskCollector.REGRESSION:
            self.model = ensemble.RandomForestRegressor(**kwargs_model)
        elif self.task in (TaskCollector.CLASSIF_BINARY, TaskCollector.CLASSIF_MULTICLASS):
            self.model = ensemble.RandomForestClassifier(**kwargs_model)
        else:
            raise ValueError()

    def fit(self, X: np.ndarray, y: np.ndarray):
        # fit model
        self.model.fit(X, y)

    def predict(self, X):
        if self.task == TaskCollector.REGRESSION:
            return self.model.predict(X)
        if self.task in (TaskCollector.CLASSIF_MULTICLASS, TaskCollector.CLASSIF_BINARY):
            return self.model.predict_proba(X)

    def get_forest_structure(self):
        tree_list = self.model.estimators_
        return get_sklearn_DT_structure(tree_list, self.task)  

    def get_base_score(self):
        return 0

    def get_n_forests(self):
        return 1


class DeepForest():

    def __init__(self, task: str, n_classes: int, forest_depth: int, n_forests: int, n_estim: int, max_depth: int, 
                tree_max_features: int=None, RF_type: str='default', n_folds: int=1, early_stopping_rounds: int=1000):

        self.n_forests = n_forests
        self.forest_depth = forest_depth
        self.task = get_task(task, n_classes)

        # Default cascade configuration
        ca_config = {"n_extra_trees": n_forests // 2, "n_RF": n_forests // 2, "max_layers": forest_depth,
                    "early_stopping_rounds": early_stopping_rounds, "keep_in_mem": True}

        if n_forests == 1:  # If there is only one forest we make it a RF
            ca_config["n_extra_trees"] = 0
            ca_config["n_RF"] = 1
        elif RF_type == "CRF":
            ca_config["n_extra_trees"] = n_forests
            ca_config["n_RF"] = 0
        elif RF_type == "RF":
            ca_config["n_extra_trees"] = 0
            ca_config["n_RF"] = n_forests

        # Create estimator configurations
        ca_estims = {"n_folds": n_folds, "n_estimators": n_estim, "max_depth": max_depth, "n_jobs": -1,
                    "max_features": tree_max_features}

        # The criterion to use for optimising the cascade estimators
        if task == 'classification':
            ca_estims["criterion"] = "gini"
        elif task == 'regression':
            ca_estims["criterion"] = "mse"
        else:
            ValueError()

        est_configs = []
        for _ in range(ca_config["n_extra_trees"]):  # Instantiating Completely Random Forests, i.e
            crf = fl_estimators.ExtraRandomForestConfig(**ca_estims)  # ExtraRandomForest with max_features set to one.
            crf.max_features = 1
            est_configs.append(crf)
        for _ in range(ca_config["n_RF"]):  # Instantiating Breiman's RF with given parameters
            rf = fl_estimators.RandomForestConfig(**ca_estims)
            est_configs.append(rf)

        if task == "regression":
            n_classes = 1

        deep_forest = fl_layers.AutoGrowingCascadeLayer(est_configs=est_configs,
            early_stopping_rounds=ca_config["early_stopping_rounds"],
            max_layers=ca_config["max_layers"], stop_by_test=False, n_classes=n_classes,
            task=task, keep_in_mem=ca_config["keep_in_mem"])

        # Building model graph
        self.model = fl_graph.Graph()
        self.model.add(deep_forest) 

    def fit(self, X, y):
        # fit model
        self.model.fit(X, y)

    def predict(self, X):
        # get predictions
        pred = self.model.transform(X)
        # average predictions of forests
        pred = np.mean(np.split(pred, self.n_forests, axis=1), axis=0)
        # return
        return pred

    def get_forest_structure(self):
        forest_layers = []
        for l in range(self.forest_depth):
            layer = pd.DataFrame()
            total_node_count = 0
            for j in range(self.n_forests):
                # AutoGrowingCascadeLayer    |list of forests      |list of trees    |list of k-folds  |RandomForest
                forest = self.model.layers[0].layer_fit_cascades[l].fit_estimators[j].fit_estimators[0].est
                tree_list = forest.estimators_
                forest_structure = get_sklearn_DT_structure(tree_list, self.task, start_node_count=total_node_count)
                forest_structure['pred_outFeature'] = j
                layer = pd.concat([layer, forest_structure])
                total_node_count += len(forest_structure)
            forest_layers.append(layer)
        
        return forest_layers

    def get_base_score(self):
        return 0

    def get_n_forests(self):
        return self.n_forests




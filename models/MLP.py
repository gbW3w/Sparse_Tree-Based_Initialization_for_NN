import logging
import math
from torch import nn
from utils.utils_data import DataLoader_RepeatedStratifiedKFold
from utils.utils import DeviceCollector, get_timer, Time
from models.tree_models import DeepForest, RandomForest, XGBoostForest
from models.DF_to_MLP import DF_to_MLP_Transformer
import torch
from typing import Callable, Dict, List, Tuple, Union
import numpy as np
import torch.utils.data as data_utils
import torch.nn.utils.prune
from models.lsuv import LSUVinit

class SparseLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(SparseLinear, self).__init__(in_features, out_features) #create linear layer
        self.is_sparse = False

    def forward(self, batch):
        if self.is_sparse:
            self.weight.data = torch.where(self.weight_update_mask, self.weight.data, torch.tensor(0, device=self.device).float())
            self.bias.data = torch.where(self.bias_update_mask, self.bias.data, torch.tensor(0, device=self.device).float())
        return super().forward(batch)

    def set_params(self, weight, bias, keep_sparse, freeze):
        if keep_sparse: #register deactivated connections
            self.is_sparse = True
            self.register_buffer('weight_update_mask', torch.logical_not(torch.isnan(weight)))
            self.register_buffer('bias_update_mask', torch.logical_not(torch.isnan(bias)))
        weight = torch.nan_to_num(weight, 0) #set weight of deactivated connections to 0
        bias = torch.nan_to_num(bias, 0)
        self.weight.data = weight #add weight
        self.bias.data = bias #add bias  """
        if freeze:
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        """self.init_uniform()
        real_weight_mask = torch.logical_not(torch.isnan(weight))
        self.weight[real_weight_mask] = weight[real_weight_mask] #add weight
        self.bias.data = bias #add bias"""

    @property
    def device(self):
        return next(self.parameters()).device    

    def init_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1)) 
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: 
            self.bias.data.uniform_(-stdv, stdv) 

class MLP(nn.Module):

    def __init__(self, in_features: int, out_features: int, hidden_features: List[int], activation: Callable[[torch.Tensor], torch.Tensor],
                final_activation:  Callable[[torch.Tensor], torch.Tensor]=None, dropout: float=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.is_init = False

        # create layers
        self.layers = nn.ModuleList()
        hidden_cur = in_features
        for hidden_next in hidden_features:
            self.layers.append(SparseLinear(hidden_cur, hidden_next))
            hidden_cur = hidden_next
        self.layers.append(SparseLinear(hidden_cur, out_features))

        self.activation = activation
        self.final_activation = final_activation
        self.logger = logging.getLogger('MLP')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # move model to device
        self.to(self.device)
        if dropout is None:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def init_uniform(self):
        self.is_init = True
        # set up timer
        timer = get_timer(self.device)
        timer.start()  
        for layer in self.layers:
            stdv = 1. / math.sqrt(layer.weight.size(1)) 
            layer.weight.data.uniform_(-stdv, stdv)
            if layer.bias is not None: 
                layer.bias.data.uniform_(-stdv, stdv)
        return timer.end()

    def init_zero(self):
        self.is_init = True
        for layer in self.layers:
            layer.weight.data.zero_()
            if layer.bias is not None: 
                layer.bias.data.zero_()

    def init_XGBoost(self, data: tuple, *, task: str, n_classes: int, max_depth: int, n_estim: int, reg_l1: float, 
                    reg_l2: float, learning_rate: float, strength01: float, strength12: float,
                    drop_layers: int=0, use_gpu: bool=False) -> XGBoostForest: 
        # set up timer
        timer = get_timer(self.device)
        timer.start()    
        # set device
        if use_gpu:
            device = DeviceCollector.GPU
        else:
            device = DeviceCollector.CPU
        # get and fit xgboost model
        model = XGBoostForest(task, n_classes, max_depth, n_estim, reg_l1, reg_l2, learning_rate, device)
        # make data dict
        data_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        data = dict((key, item) for key, item in zip(data_keys, data))
        # fit model
        model.fit(data['X_train'], data['y_train'])
        #init from model
        time = self.init_from_model(model, strength01, strength12, None, drop_layers, timer=timer)
        return model, time

    def init_RF(self, data: tuple, *, task: str, n_classes: int, max_depth: int, n_estim: int, max_features: float, strength01: float, 
                strength12: float, drop_layers: int=0) -> RandomForest:
        # set up timer
        timer = get_timer(self.device)
        timer.start()
        # get model
        model = RandomForest(task, n_classes, max_depth, n_estim, max_features)
        # make data dict
        data_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        data = dict((key, item) for key, item in zip(data_keys, data))
        # fit model
        model.fit(data['X_train'], data['y_train'])
        # init from model
        time = self.init_from_model(model, strength01, strength12, None, drop_layers, timer=timer)
        return model, time

    def init_DF(self, data: tuple, task: str, n_classes: int, forest_depth: int, n_forests: int, n_estim: int, max_depth: int, 
                strength01: float, strength12: float, strength23: float, strength_id: float, drop_layers: int=0,
                tree_max_features: float=None, RF_type: str='default', n_folds: int=1) -> DeepForest:
        # set up timer
        timer = get_timer(self.device)
        timer.start()
        # get model
        model = DeepForest(task, n_classes, forest_depth, n_forests, n_estim, max_depth, tree_max_features,
            RF_type, n_folds)
        # make data dict
        data_keys = ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']
        data = dict((key, item) for key, item in zip(data_keys, data))
        # fit model
        model.fit(data['X_train'], data['y_train'])
        # init from model
        time = self.init_from_model(model, strength01, strength12, strength23, drop_layers, strength_id, timer=timer)
        return model, time
        

    def init_from_model(self, model: Union[RandomForest, XGBoostForest, DeepForest], strength01: float, strength12: float, 
                        strength23: float, drop_layers: int, strength_id: float=None, timer: Time=None):
        # get tree structure
        forest_structure = model.get_forest_structure()
        base_score = model.get_base_score()
        n_forests = model.get_n_forests()
        # get MLP initialization parameters
        transformer = DF_to_MLP_Transformer(self.in_features, base_score, strength01, strength12, strength23, strength_id, n_forests)
        weights, biases = transformer(forest_structure)
        # return running time
        return self.initialize_weights(weights, biases, drop_layers, timer)

    def initialize_weights(self, weights: List[torch.Tensor], biases: List[torch.Tensor], drop_layers: int, timer: Time=None):
        self.is_init = True
        # get width of initialization
        init_dim = [x.size(0) if x.dim() > 0 else 1 for x in biases]
        if drop_layers > 0:
            init_dim = init_dim[:-drop_layers]
        self.logger.info(f'initialized width: {init_dim}')
        # check for dimension problems
        for i, (layer, dim) in enumerate(zip(self.layers, init_dim)):
            if layer.out_features < dim:
                raise RuntimeError(f'Layer {i} has {layer.out_features} output features but initialization requires {dim}.')
        # initialize
        dim_in = self.in_features
        for dim_out, layer, weight, bias in zip(init_dim, self.layers, weights, biases):
            new_weight = torch.full_like(layer.weight.data, np.nan)
            new_weight[:dim_out, :dim_in] = weight
            new_bias = torch.full_like(layer.bias.data, np.nan)
            new_bias[:dim_out] = bias
            #layer.weight.data[:dim_out, :dim_in] = new_weight
            #layer.bias.data[:dim_out] = new_bias
            layer.set_params(new_weight, new_bias, keep_sparse=False, freeze=False)
            dim_in = dim_out
        self.layers_initialized = len(init_dim)
        # return running time
        return timer.end()

    def init_winning_ticket(self, data, task, epochs, batch_size, learning_rate, metrics, weight_decay=0, regularize_l1=None,
        pruning_rounds=3, pruning_rate=0.2, early_stop=None):
        self.is_init = True
        self.logger.info('Starting search for winning ticket.')
        # set up timer
        timer = get_timer(self.device)
        timer.start()
        # remove pruning if already present
        for layer in self.layers:
            if torch.nn.utils.prune.is_pruned(layer):
                torch.nn.utils.prune.remove(layer, 'weight')
                torch.nn.utils.prune.remove(layer, 'bias')
        # Glorot initialization
        init_weights_per_layer = list()
        for layer in self.layers:
            init_weights = torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            init_weights_per_layer.append(init_weights)

        pruning_rate_per_round = 1 - pruning_rate ** (1 / pruning_rounds)
        for i in range(pruning_rounds):
            # fit MLP to data
            self.fit(data, task, epochs, batch_size, learning_rate, weight_decay, metrics, regularize_l1, early_stop=None)
            # prune MLP
            for layer in self.layers:
                torch.nn.utils.prune.l1_unstructured(layer, 'weight', pruning_rate_per_round)
                torch.nn.utils.prune.l1_unstructured(layer, 'bias', pruning_rate_per_round)
            # reset weights to original initialization
            for layer, init_weight in zip(self.layers, init_weights_per_layer):
                layer.weight_orig.data = init_weight
                layer.bias_orig.data = torch.zeros_like(layer.bias_orig)
            self.logger.info(f'pruning round {i+1}/{pruning_rounds}, {100 * (1 - torch.mean(layer.weight_mask))}\% of weights have been removed.') 
        self.logger.info('Winning ticket determined.')   
        return timer.end()

    def init_lsuv(self, data):
        self.is_init = True
        self.logger.info('Doing LSUV initialization...')  
        timer = get_timer(self.device)
        timer.start()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device).float()
        LSUVinit(self, data, cuda=True, verbose=False)
        self.logger.info('LSUV initialization done.')  
        return timer.end()

    def init_xavier(self):
        self.is_init = True
        # set up timer
        timer = get_timer(self.device)
        timer.start()  
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        return timer.end()

    def forward(self, x, return_layer=None):
        # check if initalization has been done
        assert self.is_init, 'MLP has not yet been initialized (please initialize explicitly)'
        # transform to torch.Tensor if necessary
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.device)

        # apply first layer
        x = self.layers[0](x)

        # other layers
        for i, layer in enumerate(self.layers[1:]):
            # apply activation function
            x = self.activation(x)
            # stop and return if return_layer reached
            if return_layer is not None and return_layer == i:
                return x
            # apply dropout layer
            if self.dropout is not None and i > 2:
                x = self.dropout(x)
            # apply next layer
            x = layer(x)
        
        # final activation
        if self.final_activation is not None:
            x = self.final_activation(x)

        return x 

    def fit(self, data: Tuple[Union[np.ndarray, torch.Tensor]], task: str, epochs: int=10, batch_size: int=256, learning_rate: float=0.01, 
            weight_decay: float=0, metrics: Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]=None, regularize_l1: float=None, early_stop: int=None):
        """
        Trains the neural network with gradient descent on the data.

        :param data: tuple, of input (X) - output (Y) data for train/dev/test set
        :param task: 'regression' or 'classification'
        """
        device = self.device
        self.logger.info(f'Start training using {device}')

        valid_tasks = {'classification', 'regression'}
        if task not in valid_tasks:
            raise ValueError('task: must be one of %r' % valid_tasks)

        # make data dict
        data_keys = ['x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test']
        data = dict((key, item) for key, item in zip(data_keys, data))

        # prepare data sets
        for key, data_set in data.items():
            # convert to torch if neccessary
            if isinstance(data_set, np.ndarray):
                data_set = torch.from_numpy(data_set)
            # transfrom to correct dtype
            if task == 'regression':
                data_set = data_set.float()
            elif task == 'classification' and 'x' in key:
                data_set = data_set.float()
            elif task == 'classification' and 'y' in key:
                data_set = data_set.long()
            # move data to device
            #data_set = data_set.to(device)
            # set to dict
            data[key] = data_set
            

        # data loaders
        train_loader = data_utils.DataLoader(
            data_utils.TensorDataset(data['x_train'], data['y_train']), 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            drop_last=True)
        val_loader = data_utils.DataLoader(
            data_utils.TensorDataset(data['x_val'], data['y_val']), 
            batch_size=batch_size*8, 
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        test_loader = data_utils.DataLoader(
            data_utils.TensorDataset(data['x_test'], data['y_test']), 
            batch_size=batch_size*8, 
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # set up metric tracking
        metrics_train_ = dict()
        metrics_val_ = dict()
        metrics_test_ = dict()
        weight_hist_ = dict()
        for name, metric in metrics.items():
            metrics_train_[name] = list()
            metrics_val_[name] = list()
            metrics_test_[name] = list()

        # metrics at initialization
        for name, metric in metrics.items():
            metrics_train_[name].append(self.eval_metric(metric, train_loader, device))
            metrics_val_[name].append(self.eval_metric(metric, val_loader, device))
            metrics_test_[name].append(self.eval_metric(metric, test_loader, device))

        # set up timing
        train_time = list()
        test_time = 0
        timer = get_timer(device)

        # set up early stopping
        loss_fun_name = list(metrics.keys())[0]
        loss_fun = metrics[loss_fun_name]
        best_val_epoch = 0
        best_val_value = metrics_val_[loss_fun_name][-1]
        best_val_epoch_weights = list()
        for layer in self.layers:
            best_val_epoch_weights.append(torch.cat([torch.flatten(layer.weight), layer.bias]))

        # save initial parameters
        init_weights = list()
        for layer in self.layers:
            init_weights.append(torch.cat([torch.flatten(layer.weight), layer.bias]))
        
        # training loop
        for epoch in range(epochs):
            self.logger.info(f'[{epoch}/{epochs}] train loss {metrics_train_[loss_fun_name][epoch]}, '
                f'val loss {metrics_val_[loss_fun_name][epoch]}, test loss {metrics_test_[loss_fun_name][epoch]}')

            # log weight histogram of initialized layers
            weight_hist = list()
            for i, layer in enumerate(self.layers):
                weights = torch.cat([torch.flatten(layer.weight), layer.bias]).detach().cpu().numpy()
                weight_hist.append(tuple(np.histogram(weights, bins=50)))
            weight_hist_[epoch] = (self.layers_initialized if hasattr(self, 'layers_initialized') else 0, weight_hist)
        
            self.train()
            timer.start()
            for batch, target in train_loader:
                batch, target = batch.to(device), target.to(device)
                pred = self(batch)
                loss_train = loss_fun(pred, target)
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()
                if regularize_l1 is not None:
                    self.soft_threshold_layers(regularize_l1)
            train_time.append(timer.end())

            self.eval()
            for name, metric in metrics.items():
                metrics_train_[name].append(self.eval_metric(metric, train_loader, device))
                metrics_val_[name].append(self.eval_metric(metric, val_loader, device))
                test_metric_aux, test_time_aux = self.eval_metric(metric, test_loader, device, measure_time=True)
                metrics_test_[name].append(test_metric_aux)
                test_time += test_time_aux/epochs

            if best_val_value is None or metrics_val_[loss_fun_name][-1] < best_val_value:
                best_val_value = metrics_val_[loss_fun_name][-1]
                best_val_epoch = epoch
                best_val_epoch_weights = list()
                for layer in self.layers:
                    best_val_epoch_weights.append(torch.cat([torch.flatten(layer.weight), layer.bias]))

            if early_stop is not None and epoch-best_val_epoch > early_stop:
                self.logger.info(f'Early stopping activated, were at epoch {epoch} and best epoch is {best_val_epoch}')
                break
                    
        self.logger.info(f'[{epoch+1}/{epochs}] train loss {metrics_train_[loss_fun_name][epoch+1]}, '
                f'val loss {metrics_val_[loss_fun_name][epoch+1]}, test loss {metrics_test_[loss_fun_name][epoch+1]}')
                    

        metrics_train_['time'] = sum(train_time[:best_val_epoch+1])
        metrics_train_['best_val_epoch'] = best_val_epoch
        metrics_test_['time'] = test_time

        # calculate weight differences before and after training
        wdiff_hist_ = list()
        for init_weight, best_val_epoch_weight in zip(init_weights, best_val_epoch_weights):
            wdiff = (init_weight - best_val_epoch_weight).detach().cpu().numpy()
            wdiff_hist = tuple(np.histogram(wdiff, bins=50))
            wdiff_hist_.append(wdiff_hist)


        # transform metrics to np arrays
        for name, metric in metrics.items():
            metrics_train_[name] = np.array(metrics_train_[name])
            metrics_val_[name] = np.array(metrics_val_[name])
            metrics_test_[name] = np.array(metrics_test_[name])

        return metrics_train_, metrics_val_, metrics_test_, weight_hist_, wdiff_hist_


    def eval_metric(self, metric, data_loader, device, measure_time=False):
        """
        Evaluates the performance of a model.

        :param model: model used for predictions
        :param metric: metric used to determine performance
        :param data_loader: iterator that provides the data that the performance is evaluated on 
            (all batches should have an equal number of observations)
        :param device: device that the model is on

        :return: float, performance of `model` on `data_loader` according to `metric`
        """
        # set up time measurement
        if measure_time:
            timer = get_timer(device)
            timer.start()

        # evaluate performance
        performance = 0
        for batch, target in data_loader:
            with torch.no_grad():
                pred = self(batch.to(device)).cpu().numpy()
            performance += metric(pred, target.numpy())*len(target)
        performance /= len(data_loader.dataset)

        # return time elapsed if needed
        if measure_time:
            test_time = timer.end()
            return performance, test_time
        return performance

    def soft_threshold_layers(self, threshold):
        for layer in self.layers:
            layer.weight.data = MLP.soft_threshold(layer.weight.data, threshold)
            layer.bias.data = MLP.soft_threshold(layer.bias.data, threshold)

    @staticmethod
    def soft_threshold(x, threshold):
        torch_zero = torch.tensor(0, dtype=torch.float32, device=x.get_device() if x.get_device() > -1 else 'cpu')
        x = torch.where(torch.abs(x) <= threshold, torch_zero, x)
        x = torch.where(x > threshold, x - threshold, x)
        x = torch.where(x < -threshold, x + threshold, x)
        return x

    def get_params(self):
        return [(layer.weight_orig.data, layer.bias_orig.data) if torch.nn.utils.prune.is_pruned(layer) else (layer.weight.data, layer.bias.data)
            for layer in self.layers]

    def set_params(self, params):
        for layer, (weight, bias) in zip(self.layers, params):
            if torch.nn.utils.prune.is_pruned(layer):
                layer.weight_orig.data = weight
                layer.bias_orig.data = bias
            else:
                layer.weight.data = weight
                layer.bias.data = bias


if __name__ == '__main__':
    data_name = ['housing', 'year', 'adult', 'bank', 'covertype', 'volkert', 'eye'][0]
    DataLoader = DataLoader_RepeatedStratifiedKFold(data_name, n_splits=5, n_repeats=1, n_max=1)
    n = DataLoader.get_n_splits()
    for data in DataLoader:
        model = MLP(DataLoader.in_features, DataLoader.out_features, 500, 3, activation=nn.Tanh())
        model.gbtree_init(data, DataLoader.task, DataLoader.n_classes, 2, 1, 0.043, 1.5e-8, 0.022, 1000, 1000, False)
        X_val = torch.from_numpy(data['X_val']).float()
        model(X_val)

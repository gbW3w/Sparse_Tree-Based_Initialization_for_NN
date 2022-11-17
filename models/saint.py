# The SAINT model.
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
#from torch import einsum
#from einops import rearrange

from models.saint_lib.models.pretrainmodel import SAINT as SAINTModel
from models.saint_lib.data_openml import DataSetCatCon
from models.saint_lib.augmentations import embed_data_mask
from utils.utils import get_timer

## Code taken and adapted from https://github.com/somepago/saint
'''
    SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
    (https://arxiv.org/abs/2106.01342)
    
    Code adapted from: https://github.com/kathrinse/TabSurvey/
'''


class SAINT():

    def __init__(self, in_features, out_features, dim, depth, heads, dropout, cat_idx, cat_dims, task):
        self.in_features = in_features
        self.out_features = out_features
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dropout = dropout
        self.cat_idx = cat_idx
        self.task = task
        if cat_idx is not None:
            num_idx = list(set(range(in_features)) - set(cat_idx))
            # Appending 1 for CLS token, this is later used to generate embeddings.
            cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)
        else:
            num_idx = list(range(in_features))
            cat_dims = np.array([1])

        # Decreasing some hyperparameter to cope with memory issues
        # we're doing this in search_spaces.py
        #dim = self.params["dim"] if num_features < 50 else 8
        #self.batch_size = self.args.batch_size if args.num_features < 50 else 64

        #print("Using dim %d and batch size %d" % (dim, self.batch_size))

        self.model = SAINTModel(
            categories=tuple(cat_dims),
            num_continuous=len(num_idx),
            dim=dim,
            dim_out=1,
            depth=self.depth,  # 6
            heads=self.heads,  # 8
            attn_dropout=self.dropout,  # 0.1
            ff_dropout=self.dropout,  # 0.1
            mlp_hidden_mults=(4, 2),
            cont_embeddings="MLP",
            attentiontype="colrow",
            final_mlp_style="sep",
            y_dim=self.out_features
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger('SAINT')

    def fit(self, data, metrics, task, batch_size=256, epochs=100, early_stop=None):
        self.task = task
        self.batch_size = batch_size

        optimizer = optim.AdamW(self.model.parameters(), lr=0.00003)

        self.model.to(self.device)

        # make data dict
        data_keys = ['x_train', 'x_val', 'x_test', 'y_train', 'y_val', 'y_test']
        data = dict((key, item) for key, item in zip(data_keys, data))

        # prepare data sets
        for key, data_set in data.items():
            # convert to torch if neccessary
            #if isinstance(data_set, np.ndarray):
            #    data_set = torch.from_numpy(data_set)
            # transfrom to correct dtype
            if task == 'regression':
                data_set = data_set.astype(np.float)
            elif task == 'classification' and 'x' in key:
                data_set = data_set.astype(np.float)
            elif task == 'classification' and 'y' in key:
                data_set = data_set.astype(np.long)
            # SAINT wants it like this...
            if 'x' in key:
                data_set = {'data': data_set, 'mask': np.ones_like(data_set)}
            if 'y' in key:
                data_set = {'data': data_set.reshape(-1, 1)}
            # set to dict
            data[key] = data_set

        # loss function
        loss_fun_name = list(metrics.keys())[0]
        loss_fun = metrics[loss_fun_name]
        # data loaders
        train_loader = data_utils.DataLoader(
            DataSetCatCon(data['x_train'], data['y_train'], self.cat_idx, task),
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            drop_last=True)
        val_loader = data_utils.DataLoader(
            DataSetCatCon(data['x_val'], data['y_val'], self.cat_idx, task), 
            batch_size=batch_size*10, 
            shuffle=False,
            pin_memory=True,
            drop_last=False)
        test_loader = data_utils.DataLoader(
            DataSetCatCon(data['x_test'], data['y_test'], self.cat_idx, task), 
            batch_size=batch_size*10, 
            shuffle=False,
            pin_memory=True,
            drop_last=False)

        # set up metric tracking
        metrics_train_ = dict()
        metrics_val_ = dict()
        metrics_test_ = dict()
        for name, metric in metrics.items():
            metrics_train_[name] = list()
            metrics_val_[name] = list()
            metrics_test_[name] = list()

        # metrics at initialization
        for name, metric in metrics.items():
            metrics_train_[name].append(self.eval_metric(metric, train_loader, self.device))
            metrics_val_[name].append(self.eval_metric(metric, val_loader, self.device))
            metrics_test_[name].append(self.eval_metric(metric, test_loader, self.device))

        # set up timing
        train_time = list()
        test_time = 0
        timer = get_timer(self.device)

        # set up early stopping
        best_val_epoch = 0
        best_val_value = metrics_val_[loss_fun_name][-1]

        for epoch in range(epochs):
            self.logger.info(f'[{epoch}/{epochs}] train loss {metrics_train_[loss_fun_name][epoch]}, '
                f'val loss {metrics_val_[loss_fun_name][epoch]}, test loss {metrics_test_[loss_fun_name][epoch]}')
            
            self.model.train()
            timer.start()
            for i, data in enumerate(train_loader, 0):
                optimizer.zero_grad()

                # x_categ is the the categorical data,
                # x_cont has continuous data,
                # y_gts has ground truth ys.
                # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS
                # token) set to 0s.
                # con_mask is an array of ones same shape as x_cont.
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                # We are converting the data to embeddings in the next step
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)

                reps = self.model.transformer(x_categ_enc, x_cont_enc)

                # select only the representations corresponding to CLS token
                # and apply mlp on it in the next step to get the predictions.
                y_reps = reps[:, 0, :]

                y_outs = self.model.mlpfory(y_reps)

                if task == "regression":
                    y_gts = y_gts.to(self.device)
                elif task == "classification":
                    y_gts = y_gts.to(self.device).squeeze()
                else:
                    y_gts = y_gts.to(self.device).float()

                loss = loss_fun(y_outs, y_gts)
                loss.backward()
                optimizer.step()
            train_time.append(timer.end())

            for name, metric in metrics.items():
                metrics_train_[name].append(self.eval_metric(metric, train_loader, self.device))
                metrics_val_[name].append(self.eval_metric(metric, val_loader, self.device))
                test_metric_aux, test_time_aux = self.eval_metric(metric, test_loader, self.device, measure_time=True)
                metrics_test_[name].append(test_metric_aux)
                test_time += test_time_aux/epochs

            if best_val_value is None or metrics_val_[loss_fun_name][-1] < best_val_value:
                best_val_value = metrics_val_[loss_fun_name][-1]
                best_val_epoch = epoch

            if early_stop is not None and epoch-best_val_epoch > early_stop:
                self.logger.info(f'Early stopping activated, were at epoch {epoch} and best epoch is {best_val_epoch}')
                break
                    
        self.logger.info(f'[{epoch+1}/{epochs}] train loss {metrics_train_[loss_fun_name][epoch+1]}, '
            f'val loss {metrics_val_[loss_fun_name][epoch+1]}, test loss {metrics_test_[loss_fun_name][epoch+1]}')

        metrics_train_['time'] = sum(train_time[:best_val_epoch+1])
        metrics_train_['best_val_epoch'] = best_val_epoch
        print(train_time[:best_val_epoch+1], metrics_train_['time'], metrics_train_['best_val_epoch'])
        metrics_test_['time'] = test_time

        # transform metrics to np arrays
        for name, metric in metrics.items():
            metrics_train_[name] = np.array(metrics_train_[name])
            metrics_val_[name] = np.array(metrics_val_[name])
            metrics_test_[name] = np.array(metrics_test_[name])
        return metrics_train_, metrics_val_, metrics_test_ 

    def eval_metric(self, metric, data_loader, device, measure_time=False):
        # set up time measurement
        if measure_time:
            timer = get_timer(device)
            timer.start()

        # evaluate performance
        performance = 0.0
        self.model.eval()
        with torch.no_grad():
            for data in data_loader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                if self.task == "regression":
                    y_gts = y_gts
                elif self.task == "classification":
                    y_gts = y_gts.squeeze()
                else:
                    y_gts = y_gts.float()

                performance += metric(y_outs.cpu().numpy(), y_gts.numpy())*len(y_outs)
            performance /= len(data_loader.dataset)

        # return time elapsed if needed
        if measure_time:
            test_time = timer.end()
            return performance, test_time
        return performance

    def predict(self, X):
        if self.task == "regression":
            self.predictions = self.predict_helper(X,)
        else:
            self.predict_proba(X)
            self.predictions = np.argmax(self.prediction_probabilities, axis=1)

        return self.predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_helper(X, 'classification')

        # If binary task returns only probability for the true class, adapt it to return (N x 2)
        if probas.shape[1] == 1:
            probas = np.concatenate((1 - probas, probas), 1)

        self.prediction_probabilities = probas
        return self.prediction_probabilities

    def predict_helper(self, X):
        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': np.ones((X['data'].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.cat_idx, self.task)
        testloader = data_utils.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()

        predictions = []

        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)

                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                y_reps = reps[:, 0, :]
                y_outs = self.model.mlpfory(y_reps)

                if self.args.objective == "binary":
                    y_outs = torch.sigmoid(y_outs)
                elif self.args.objective == "classification":
                    y_outs = F.softmax(y_outs, dim=1)

                predictions.append(y_outs.detach().cpu().numpy())
        return np.concatenate(predictions)

    """def attribute(self, X, y, strategy=""):
        #Generate feature attributions for the model input.
        #    Two strategies are supported: default ("") or "diag". The default strategie takes the sum
        #    over a column of the attention map, while "diag" returns only the diagonal (feature attention to itself)
        #    of the attention map.
        #    return array with the same shape as X.
        
        global my_attention
        # self.load_model(filename_extension="best", directory="tmp")

        X = {'data': X, 'mask': np.ones_like(X)}
        y = {'data': np.ones((X['data'].shape[0], 1))}

        test_ds = DataSetCatCon(X, y, self.args.cat_idx, self.args.objective)
        testloader = DataLoader(test_ds, batch_size=self.args.val_batch_size, shuffle=False, num_workers=4)

        self.model.eval()
        # print(self.model)
        # Apply hook.
        my_attention = torch.zeros(0)

        def sample_attribution(layer, minput, output):
            global my_attention
            # print(minput)
            #an hook to extract the attention maps. 
            h = layer.heads
            q, k, v = layer.to_qkv(minput[0]).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
            sim = einsum('b h i d, b h j d -> b h i j', q, k) * layer.scale
            my_attention = sim.softmax(dim=-1)

        # print(type(self.model.transformer.layers[0][0].fn.fn))
        self.model.transformer.layers[0][0].fn.fn.register_forward_hook(sample_attribution)
        attributions = []
        with torch.no_grad():
            for data in testloader:
                x_categ, x_cont, y_gts, cat_mask, con_mask = data

                x_categ, x_cont = x_categ.to(self.device), x_cont.to(self.device)
                cat_mask, con_mask = cat_mask.to(self.device), con_mask.to(self.device)
                # print(x_categ.shape, x_cont.shape)
                _, x_categ_enc, x_cont_enc = embed_data_mask(x_categ, x_cont, cat_mask, con_mask, self.model)
                reps = self.model.transformer(x_categ_enc, x_cont_enc)
                # y_reps = reps[:, 0, :]
                # y_outs = self.model.mlpfory(y_reps)
                if strategy == "diag":
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].diagonal(0, 1, 2))
                else:
                    attributions.append(my_attention.sum(dim=1)[:, 1:, 1:].sum(dim=1))

        attributions = np.concatenate(attributions)
        return attributions"""
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
from torch import logit, nn

#def sigmoid(x):
#    if isinstance(x, np.ndarray):
#        return 1 / (1 + np.exp(-x))
#    else:
#        return 1 / (1 + torch.exp(-x))

#def transform_logits(x):
#    out_features = x.shape[1]
#    if out_features == 1: #binary classification
#        return sigmoid(x)
#    raise ValueError()

def softmax(x: np.ndarray):
    if x.ndim == 2:
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    elif x.ndim == 1:
        return np.exp(x)/np.sum(np.exp(x))
    raise ValueError('`x` must be either 1d or 2d.')

class Accuracy():
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, x, y):
        if x.ndim != 2:
            raise ValueError()
        if self.logits:
            x = softmax(x)
        #if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        #    if n_classes == 1: #binary classification
        #        x = np.concatenate([1-x, x], axis=1)
        return np.sum(np.argmax(x, axis=1) == y)/float(y.shape[0])
        #if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        #    if n_classes == 1: #binary classification
        #        x = torch.concat([1-x, x], dim=1)
        #    return torch.sum(torch.argmax(x, axis=1) == y)/float(y.size(0))
        #raise TypeError()

"""def confusion_matrix_binary(x, y, logits=True):
    if logits:
        x = softmax(x)
    #if isinstance(x, torch.Tensor):
    #    x = x.detach().cpu().numpy()
    #if isinstance(y, torch.Tensor):
    #    y = y.detach().cpu().numpy()
    pred = (x > 0.5).astype(np.int64)
    return confusion_matrix(y, pred)/y.size

def confusion_matrix_binary_withLogits(x, y):
    return confusion_matrix_binary(x, y, logits=True)"""

class AurocScoreBinary():
    def __init__(self, logits):
        self.logits = logits

    def __call__(self, x, y):
        if x.shape[1] != 2:
            raise ValueError('Input is not a binary calssification problem.')
        if self.logits:
            x = softmax(x)
        #if isinstance(x, torch.Tensor):
        #    x = x.detach().cpu().numpy()
        #if isinstance(y, torch.Tensor):
        #    y = y.detach().cpu().numpy()
        try:
            return roc_auc_score(y, x[:,1])
        except ValueError:
            print('AUROC calculation failed due to a ValueError!')
            return 0


class MSELoss():

    def __init__(self):
        pass

    def __call__(self, x, y):
        x = x.squeeze()
        y = y.squeeze()
        assert x.shape == y.shape
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            return np.mean(np.square(x - y))
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            return nn.functional.mse_loss(x, y)
        raise ValueError()

class CrossEntropyLoss():

    def __init__(self, class_weights: np.ndarray, device: torch.device, logits: bool):
        self.class_weights = torch.from_numpy(class_weights).float()
        self.logits = logits

    def __call__(self, x, y):
        is_numpy = False
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            is_numpy = True
        #out_features = x.size(1)
        #if out_features == 1: #binary classification
        #    if logits:
        #        loss = nn.functional.binary_cross_entropy_with_logits(x.squeeze().float(), y.float())
        #    else:
        #        loss = nn.functional.binary_cross_entropy(x.squeeze().float(), y.float())
        #else: #multi-class classification
        class_weights = self.class_weights.to(x)
        if self.logits:
            loss = nn.functional.cross_entropy(x, y, weight=class_weights)
        else:
            probas = torch.gather(x, 1, torch.unsqueeze(y, dim=1)).squeeze()
            weights = torch.gather(class_weights, 0, y)
            loss = -torch.mean(torch.log(probas)*weights)
        if is_numpy:
            loss = loss.numpy()
        return loss
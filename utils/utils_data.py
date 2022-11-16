import numpy as np
import logging
from sklearn import preprocessing
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from utils.load_datasets import *
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
import sys
import os

class DataLoader_RepeatedStratifiedKFold():

    def __init__(self, data_name, n_splits, n_repeats, one_hot, n_max=None):

        # for data set
        X, y, is_categorical, in_features, out_features, task, n_classes = DataLoader_RepeatedStratifiedKFold.load_data(data_name)
        self.X = X
        self.y = y
        self.is_categorical = is_categorical
        self.is_numerical = [not x for x in is_categorical]
        self.in_features = in_features
        self.out_features = out_features
        self.task = task
        self.n_classes = n_classes
        self.n_max = n_max
        self.data_name = data_name
        self.class_weights = None

        # for RepeatedStratifiedKFold
        if self.task == 'regression':
            self.rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats)
        if self.task == 'classification':
            self.rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

        # for data normalization
        self.standard_scaler = preprocessing.StandardScaler()

        # One-hot-encoding
        if one_hot:
            dummies = list()
            for col in self.X[:,self.is_categorical].T:
                dummies.append(pd.get_dummies(col).values)
            self.X = np.concatenate([self.X[:, self.is_numerical], *dummies], axis=1)
            self.in_features = self.X.shape[1]
            nb_num_features = sum(self.is_numerical)
            self.is_categorical = [False]*nb_num_features + [True]*(self.in_features-nb_num_features)
            self.is_numerical = [not x for x in self.is_categorical]

    def __iter__(self):
        self.index_iter = self.rkf.split(self.X, self.y)
        self.n_count = 0
        return self

    def __next__(self):

        if self.n_max is not None and self.n_count >= self.n_max:
            raise StopIteration

        index_train, index_test = next(self.index_iter)

        X_train, X_test = self.X[index_train], self.X[index_test]
        y_train, y_test = self.y[index_train], self.y[index_test]
        if self.task == 'regression':
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)

        # Data normalization
        X_train[:,self.is_numerical] = self.standard_scaler.fit_transform(X_train[:,self.is_numerical])
        X_val[:,self.is_numerical] = self.standard_scaler.transform(X_val[:,self.is_numerical])
        X_test[:,self.is_numerical] = self.standard_scaler.transform(X_test[:,self.is_numerical])

        self.n_count += 1

        return {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_train': y_train, 'y_val': y_val, 'y_test': y_test}

    def get_n_splits(self):
        if self.n_max is not None:
            return min(self.n_max, self.rkf.get_n_splits())
        return self.rkf.get_n_splits()

    def get_class_weights(self):
        if self.class_weights is None:
            return np.ones(self.n_classes)/self.n_classes
        return self.class_weights

    def set_class_weights(self, beta: float):
        if self.task == 'regression':
            raise ValueError('class weights can only be calculated for classification problems.')
        n_samples = np.bincount(self.y)
        assert n_samples.size == self.n_classes, 'Bincount problem.'
        self.class_weights = (1-beta)/(1-np.power(beta, n_samples))
        self.class_weights /= np.sum(self.class_weights)
        print('set class weights', self.class_weights)

    @staticmethod
    def load_data(data):
        n_class = 1
        if data == 'protein':
            protein_path = "../datasets/protein/"
            X, y, is_categorical = load_protein(protein_path)
            in_features = 9
            out_features = 1
            task = 'regression'

        elif data == 'airbnb':
            airbnb_path = "../datasets/airbnb_berlin/"
            # Regression Airbnb
            X, y, is_categorical = load_airbnb(airbnb_path)
            in_features = 13
            out_features = 1
            task = 'regression'

        elif data == 'letter':
            raise NotImplementedError()
            X_train, Y_train, x_test, y_test = load_letter()
            in_features = 16
            out_features = 26
            task = 'classification'
            error = 'count'
            n_class = 26

        elif data == 'adult':
            adult_path = "../datasets/adult/"
            
            X, y, is_categorical = load_adult(adult_path)
            in_features = X.shape[1]
            out_features = 2
            task = 'classification'
            n_class = 2

        elif data == 'higgs':
            path = "../datasets/higgs/"
            X, y, is_categorical = load_higgs(path)
            in_features = 28
            out_features = 2
            task = 'classification'
            n_class = 2
            

        elif data == "yeast":
            raise NotImplementedError()
            X_train, Y_train, x_test, y_test = load_yeast()
            in_features = 8
            out_features = 10
            task = 'classification'
            error = 'count'
            n_class = 10

        elif data == 'uniform':
            raise NotImplementedError()
            d, n = 2, 1000
            np.random.seed(1)
            X_train = np.random.uniform(size=(n,d))
            Y_train = np.exp(3*np.sum(X_train, axis=1))
            x_test = np.random.uniform(size=(n,d))
            y_test = np.exp(3*np.sum(x_test, axis=1))
            in_features = d
            out_features = 1
            task = 'regression'
            error = 'mse'

        elif data == 'housing':
            data = fetch_california_housing()
            X, y = data['data'], data['target']
            is_categorical = [False] * 8
            in_features = 8
            out_features = 1
            task = 'regression'

        elif data == 'covertype':
            covertype_path = "../datasets/covertype/"
            
            X, y, is_categorical = load_covertype(covertype_path)
            in_features = X.shape[1]
            out_features = 7
            task = 'classification'
            n_class = out_features


        elif data == 'bank':
            bank_path = "../datasets/bank/"
            
            X, y, is_categorical = load_bank(bank_path)
            in_features = X.shape[1]
            out_features = 2
            task = 'classification'
            n_class = 2

        elif data == 'volkert':
            volkert_path = '../datasets/volkert/'
            X, y, is_categorical = load_volkert(volkert_path)
            in_features = X.shape[1]
            out_features = 10
            task = 'classification'
            n_class = out_features

        elif data == 'heloc':
            heloc_path = "../datasets/HELOC/"
            
            X, y, is_categorical = load_heloc(heloc_path)
            in_features = X.shape[1]
            out_features = 2
            task = 'classification'
            n_class = 2

        elif data == 'blastchar':
            path = "../datasets/blastchar/"
            
            X, y, is_categorical = load_blastchar(path)
            in_features = X.shape[1]
            out_features = 2
            task = 'classification'
            n_class = 2
            
        elif data == 'diamonds':
            path = "../datasets/diamonds/"
            
            X, y, is_categorical = load_diamonds(path)
            in_features = X.shape[1]
            out_features = 1
            task = 'regression'


        else:
            raise ValueError()

        return X, y, is_categorical, in_features, out_features, task, n_class

def handle_loggers(file=None, terminal=False, level='debug'):
    root = logging.getLogger()
    if level == 'info':
        root.setLevel(logging.INFO)
    else:
        root.setLevel(logging.DEBUG)
    root.handlers = []
    if terminal:
        handler = logging.StreamHandler(sys.stdout)
        root.addHandler(handler)
    if file is not None:
        handler = logging.FileHandler(file, 'w')
        root.addHandler(handler)
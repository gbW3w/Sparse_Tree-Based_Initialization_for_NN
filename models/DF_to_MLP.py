from grpc import stream_stream_rpc_method_handler
from kiwisolver import strength
import pandas as pd
import torch
import numpy as np
import copy
from typing import List, Tuple, Union


class DF_to_MLP_Transformer():

    def __init__(self, in_features:int, base_score: float, strength01: float, strength12: float, strength23: float, strength_id: float, n_forests: int):
        self.DF_in_features = in_features
        self.base_score = base_score
        self.strength01 = strength01
        self.strength12 = strength12
        self.strength23 = strength23
        self.strength_id = strength_id
        self.n_forests = n_forests

    def __call__(self, DF_structure: Union[pd.DataFrame, List[pd.DataFrame]]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        # make list if single forest
        if isinstance(DF_structure, pd.DataFrame):
            DF_structure = [DF_structure]

        # get number of DF layers
        n_layers = len(DF_structure)

        # initiate parameter lists
        weights = [] 
        biases = []

        in_features = self.DF_in_features
        for i, forest_structure in enumerate(DF_structure):
            first_DF_layer = (i == 0)
            last_DF_layer = (i == n_layers - 1)
            forest_weights, forest_biases, in_features = self.forest_to_MLP_parameters(forest_structure, in_features, first_DF_layer, last_DF_layer)
            weights += forest_weights
            biases += forest_biases

        if self.n_forests > 1:
            mean_layer_weight, mean_layer_bias = self.init_mean_layer(in_features)
            weights.append(mean_layer_weight)
            biases.append(mean_layer_bias)

        return weights, biases


    def forest_to_MLP_parameters(self, forest_structure: pd.DataFrame, in_features: int, first_DF_layer: bool, last_DF_layer: bool):   
        W01, b01 = self.init_firstLayer(forest_structure, in_features, first_DF_layer, last_DF_layer)
        W12, b12 = self.init_secondLayer(forest_structure, last_DF_layer)
        W23, b23, out_features = self.init_thirdLayer(forest_structure, last_DF_layer)
        return [W01, W12, W23], [b01, b12, b23], out_features

    def init_firstLayer(self, forest_structure: pd.DataFrame, in_features: int, first_DF_layer: bool, last_DF_layer: bool):
        """
        Returns the weights of the first layer of a DF MLP-encoding.
        """
        # get number of inner nodes in forest
        nb_inner = np.sum(~forest_structure['is_leaf'])

        # get number of identity and output features
        id_features_in = 0 if first_DF_layer else self.DF_in_features
        id_features_out = 0 if last_DF_layer else self.DF_in_features
        out_features = nb_inner + id_features_out

        # instantiate weight and bias tensors
        W01 = torch.full((out_features, in_features), np.nan)
        b01 = torch.full((out_features,), np.nan)

        # get id inner mapping
        id_inner_mapping, _ = get_idInnerLeafMapping(forest_structure)

        # fill weight and bias with identity
        for i in range(id_features_out):
            W01[i,i] = 1
            b01[i] = 0

        # fill weight and bias with forest
        for _, node in forest_structure.iterrows():
            # if the node is not a leaf
            if not node['is_leaf']:
                # get inner node id of current node
                id_inner = id_inner_mapping[node['id']]
                # add the number of identity features
                idx_out = id_inner + id_features_out
                # add corresponding weight and bias for first layer
                W01[idx_out, node['feature']] = self.strength01
                b01[idx_out] = -node['threshold']*self.strength01

        # rescale inputs
        if not first_DF_layer:
            # rescale the inputs that originate from previous forests 
            W01[:,id_features_in:] *= self.strength23
            # rescale the inputs that originate from identity
            W01[:,:id_features_in] *= self.strength_id
        
        # rescale outputs
        if not last_DF_layer:
             # rescale the outputs that correspond to the identity mapping (bias = 0)
            W01[:id_features_out,:] /= self.strength_id       

        return W01, b01

    def init_secondLayer(self, forest_structure: pd.DataFrame, last_DF_layer: bool):
        # get forest paths
        forest_paths = get_forest_paths(forest_structure)

        id_features = 0 if last_DF_layer else self.DF_in_features

        # get number of inner and leaf nodes in forest
        nb_inner = np.sum(~forest_structure['is_leaf'])
        nb_leaf = np.sum(forest_structure['is_leaf'])

        # get number of in and out features
        in_features = nb_inner + id_features
        out_features = nb_leaf + id_features

        # instantiate weight and bias tensors
        W12 = torch.full((out_features, in_features), np.nan)
        b12 = torch.full((out_features,), np.nan)

        # fill identity mapping
        for i in range(id_features):
            W12[i,i] = 1
            b12[i] = 0

        for path in forest_paths:
            # get input and output indices
            idx_in = [x + id_features for x in path.inner_nodes]
            idx_out = path.leaf_node + id_features
            # fill weight and bias
            W12[idx_out, idx_in] = self.strength12*torch.Tensor(path.dirs)
            b12[idx_out] = self.strength12 * (-len(path.inner_nodes) + 0.5)

        return W12, b12

    def init_thirdLayer(self, forest_structure: pd.DataFrame, last_DF_layer: bool):

        id_features = 0 if last_DF_layer else self.DF_in_features

        # get leaf id mapping
        _, id_leaf_mapping = get_idInnerLeafMapping(forest_structure)
        leaf_id_mapping = {value: key for key, value in id_leaf_mapping.items()}

        # get number of leafs and number of in- and output features
        n_leaf = np.sum(forest_structure['is_leaf']) #number of leafs in tree
        leaf_outFeatures = max([len(x) if isinstance(x, list) else 1 for x in forest_structure['leaf_value']]) #number of output feature per leaf
        tree_outFeatures = forest_structure['pred_outFeature'].nunique() #number of distinc predition features of the forest, each of which has leaf_outFeatures features itself
        in_features = n_leaf + id_features #number of MLP layer input features
        out_features = tree_outFeatures * leaf_outFeatures + id_features #number of MLP layer output features

        # instantiate weight tensor
        W23 = torch.full((out_features, in_features), np.nan) 
        b23 = torch.full((out_features,), np.nan)   

        # fill parameters with identity mapping
        for i in range(id_features):
            W23[i,i] = 1
            b23[i] = 0

        # fill parameters with forest
        for i in range(n_leaf): 
            # get out feature and leaf value
            pred_outFeature = forest_structure.loc[forest_structure['id'] == leaf_id_mapping[i], 'pred_outFeature'].item()
            leaf_value = forest_structure.loc[forest_structure['id'] == leaf_id_mapping[i], 'leaf_value'].item() #leaf value can be a list
            # get input and output indices
            idx_in = i + id_features
            idx_out = pred_outFeature*leaf_outFeatures + id_features
            # fill values
            W23[idx_out:idx_out+leaf_outFeatures,idx_in] = torch.tensor(leaf_value)/2
        b23[id_features:] = torch.sum(W23[id_features:, id_features:], dim=1) + self.base_score

        # rescale forest output
        if self.n_forests > 1 or not last_DF_layer:
            W23[id_features:,id_features:] /= self.strength23
            b23[id_features:] /= self.strength23

        return W23, b23, out_features

    def init_mean_layer(self, in_features):
        
        # instantiate parameter tensors
        if in_features % self.n_forests > 0:
            raise RuntimeError(f'Number of classes can not be deduced: {in_features} input features and {self.n_forests} forests.')
        out_features = in_features//self.n_forests
        weight = torch.zeros((out_features, in_features))
        bias = torch.zeros(out_features)

        # fill parameters
        for idx_out in range(out_features):
            idx_in = np.arange(self.n_forests)*out_features + idx_out
            weight[idx_out, idx_in] = self.strength23/self.n_forests
            bias[idx_out] = 0

        return weight, bias


class TreePath(object):
    def __init__(self, node: int):
        # inner nodes encountered during path, including root but excluding leaf
        self.inner_nodes = []
        # leaf node of the path, ie. its last node
        self.leaf_node = node
        # direction taken at each inner node (-1 is left, 1 is right)
        self.dirs = []

    def add_node(self, node: int, dir: int):
        self.inner_nodes.append(self.leaf_node)
        self.dirs.append(dir)
        self.leaf_node = node

    def __str__(self):
        return self.inner_nodes.__str__() + ' ' + self.leaf_node.__str__() + '\n' + self.dirs.__str__()

    def __lt__(self, other):
        return self.leaf_node < other.leaf_node

    def translate_ids(self, id_inner_mapping, id_leaf_mapping):
        self.inner_nodes = [id_inner_mapping[inner_node] for inner_node in self.inner_nodes]
        self.leaf_node = id_leaf_mapping[self.leaf_node]


def get_idInnerLeafMapping(forest_structure: pd.DataFrame):
    # get ids of inner nodes
    id_inner_nodes = forest_structure['id'][~forest_structure['is_leaf']]
    # map xgb ids of inner nodes to contiguous ids starting at 0
    id_inner_mapping = {item:i for i, item in enumerate(id_inner_nodes)}
    # get xgb ids of leaf nodes
    id_leaf_nodes = forest_structure['id'][forest_structure['is_leaf']]
    # map xgb ids of leaf nodes to contiguous ids starting at 0
    id_leaf_mapping = {item:i for i, item in enumerate(id_leaf_nodes)}

    return id_inner_mapping, id_leaf_mapping


def get_forest_paths(forest_structure: pd.DataFrame):

    ##### CONSTRUCT PATHS
    # initialize path with root nodes
    set_id = set(forest_structure['id'])
    set_childs = set(pd.concat([forest_structure['left_child'], forest_structure['right_child']]).dropna())
    forest_paths = [TreePath(root_node) for root_node in set_id.difference(set_childs)]
    
    i = 0
    while i < len(forest_paths):
        # get current node id
        current_node = forest_structure.loc[forest_structure['id'] == forest_paths[i].leaf_node]
        
        if current_node['is_leaf'].item():
            # evaluate next path in list
            i = i + 1
        else: 
            # append split path for right child to list
            forest_paths.append(copy.deepcopy(forest_paths[i]))
            # add left child to current path
            forest_paths[i].add_node(current_node['left_child'].item(), -1)
            # add right child to split path
            forest_paths[-1].add_node(current_node['right_child'].item(), 1)

    # sort forest paths to get them in ascending leaf index order
    forest_paths.sort()

    ##### TRANSLATE IDs TO INNER AND LEAF IDs
    # get id inner and leaf mapping
    id_inner_mapping, id_leaf_mapping = get_idInnerLeafMapping(forest_structure)
    # translate ids
    for i in range(len(forest_paths)):
        forest_paths[i].translate_ids(id_inner_mapping, id_leaf_mapping)

    return forest_paths
    
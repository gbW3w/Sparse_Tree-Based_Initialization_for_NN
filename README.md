# Sparse Tree-Based Initialization for NN
This repository contains the code that was used to produce the experimental results in "Sparse Tree-Based Initalizations for Neural Networks".

If you want to check out the code for the different intialization methods, you might want to have look at `models/MLP.py`; if you want to check out the transition of tree-based methods into MLP, you might want to have a look at `models/DF_to_MLP.py`. This ReadMe will give instructions on how to repoduce the main results of the paper.

## Hyper-parameter search
Optimal hyper parameters (HP) can be determined by calling `HP_search.py`. Please run `HP_search.py -h` for a list of required and optional arguments. An example call executing the HP search for a randomly initialized MLP on the Housing data set as we perfromed it in the paper would be

```HP_search.py --model MLP_rand_init --data housing --n_max 1 --n_trials 100```

Our code will run on one GPU if available. Note that we provide all optimal HP settings that we determined in the folder `HP_setting`, so one can skip this time consuming step. The `-out` argument can be used to generate new HP configurations without overwriting the provided ones.

Further, note that the HP of all models based on tree-based initializations are determined in TWO STEPS. In a first step, the HP of the tree-based initializer are determined. In a second step, the HP of MLP itself and of the tree-method-to-MLP-translation are determined. In accordance with the procedure in our paper, this would be done in the following way

```HP_search.py --model RF_init --data housing --n_max 1 --n_trials 25```

```HP_search.py --model MLP_RF_init --data housing --n_max 1 --n_trials 75```

## Model evaluation
Once optimal HP for a model have been determined, this model can be evaluated by calling `eval_model.py`. Please run `eval_model.py -h` for a list of required and optional arguments. An example call for the evaluation of the randomly initialized MLP on the Housing data set as we perfromed it in the paper would be

```model_eval.py --model MLP_rand_init --data housing```

## Reading the evaluation file
The results of an evaluation are stocked in a `.pkl` file. Once the evaluation is completed, one can easily print the performance of a model by calling `print_results.py`. Please run `print_results.py -h` for a list of required and optional arguments. An example call for printing the evaluation results of the randomly initialized MLP on the Housing data set would be

```printing_results.py --model MLP_rand_init --data housing```

Note that the `.pkl` files are heavy as they contain other information beyond the performance of a model (e.g., its weight distribution per layer, etc.). However, we do not provide efficient ways to extract this information yet.

## Some possible argument values
Below, we list all possible values for the arguments `--model` and `--data`.

For models: `RF` (Random Forest), `XGB` (XGBoost), `DF` (Deep Forest), `RF_init`, `XGB_init`, `DF_init`, `MLP_rand_init`, `MLP_xavier_init`, `MLP_LUVS_init`, `MLP_WT_init` (MLP with winning ticket pruning), `MLP_RF_init`, `MLP_XGB_init` (MLP GBDT init.), `MLP_DF_init`, `SAINT`

For data sets: `housing`, `airbnb`, `diamonds`, `adult`, `bank`, `blastchar`, `heloc`, `higgs`, `covertype`, `volkert`.
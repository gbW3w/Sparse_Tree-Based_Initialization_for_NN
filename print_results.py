import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import argparse

def extend_to_length(x: np.ndarray, length: int):
        new_x = np.full(length, np.nan)
        new_x[:x.size] = x
        return new_x

def print_results(data_name, method, in_suffix):
    ####### Load data
    source_file = f'eval/eval_{data_name}_{method}_{in_suffix}'
    with open(source_file+'.pkl', 'rb') as f:
        data = pickle.load(f)

    ####### Print best values
    early_stopping = True
    metrics_dict = dict()
    if early_stopping:
        choose_loss_fun = np.min
        choose_score_fun = np.max
        choose_loss_fun_arg = np.argmin
        choose_score_fun_arg = np.argmax
    else:
        choose_loss_fun = lambda x: x[-1]
        choose_score_fun = lambda x: x[-1]
        choose_loss_fun_arg = lambda x: len(x)-1
        choose_score_fun_arg = lambda x: len(x)-1

    es_string = 'Best' if early_stopping else 'Final'
    print('Method:', method)
    print('HP setting:', data['hp'])
    metrics_names = data['metrics_names']
    for mode, metrics in (('train', data['metrics_train_']), ('val', data['metrics_val_']), ('test', data['metrics_test_'])):
        for i, metric_name in enumerate(metrics_names):
            if i == 0: #it's a loss metric (best value is lowest)
                loss_mean = np.mean([choose_loss_fun(metric[metric_name]) for metric in metrics])
                loss_std = np.std([choose_loss_fun(metric[metric_name]) for metric in metrics])
                print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format(es_string, method, mode, metric_name, loss_mean, loss_std))
            else: #it's a score metric (best value is highest)
                score_mean = np.mean([choose_score_fun(metric[metric_name]) for metric in metrics])
                score_std = np.std([choose_score_fun(metric[metric_name]) for metric in metrics])
                print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format(es_string, method, mode, metric_name, score_mean, score_std))
    if early_stopping and list(data['metrics_val_'][0].values())[0].size > 1:
        for i, metric_name in enumerate(metrics_names):
            if i == 0: #it's a loss metric (best value is lowest)
                best_it = np.array([choose_loss_fun_arg(metric[metric_name]) for metric in data['metrics_val_']])
                loss_mean = np.mean([metric[metric_name][it] for metric, it in zip(data['metrics_test_'], best_it)])
                loss_std = np.std([metric[metric_name][it] for metric, it in zip(data['metrics_test_'], best_it)])
                print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format('Early stopping', method, '', metric_name, loss_mean, loss_std))
            else:
                best_it = np.array([choose_score_fun_arg(metric[metric_name]) for metric in data['metrics_val_']])
                score_mean = np.mean([metric[metric_name][it] for metric, it in zip(data['metrics_test_'], best_it)])
                score_std = np.std([metric[metric_name][it] for metric, it in zip(data['metrics_test_'], best_it)])
                print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format('Early stopping', method, '', metric_name, score_mean, score_std))


    if 'metrics_add_' in data.keys():
        # additional metrics can potentially have different metrics than train, val and test
        metrics_add_names = list(data['metrics_add_'][0].keys())
        for i, metric_name in enumerate(metrics_add_names):
            mode = 'add'
            metrics = data['metrics_add_']
            if i == 0: #it's a loss metric (best value is lowest)
                loss_mean = np.mean([choose_loss_fun(metric[metric_name]) for metric in metrics])
                loss_std = np.std([choose_loss_fun(metric[metric_name]) for metric in metrics])
                print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format(es_string, method, mode, metric_name, loss_mean, loss_std))
            else: #it's a score metric (best value is highest)
                score_mean = np.mean([choose_score_fun(metric[metric_name]) for metric in metrics])
                score_std = np.std([choose_score_fun(metric[metric_name]) for metric in metrics])
                print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format(es_string, method, mode, metric_name, score_mean, score_std))

    if 'MLP' in method or 'SAINT' in method:
        for mode, metrics in (('train', data['metrics_train_']), ('test', data['metrics_test_'])):
            print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format(es_string, method, mode, 'time', metrics[0]['time'], 0))
            if 'best_val_epoch' in metrics[0].keys():
                print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format(es_string, method, mode, 'best_val_epoch', metrics[0]['best_val_epoch'], 0))
        if 'init_time_' in data.keys():
            print(u'{0:14} {1:13} {2:5} {3:28}: {4:20} \u00B1 {5:20}'.format(es_string, method, mode, 'init_time', np.mean(data['init_time_']), 0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'eval_model',
                    description = 'Evaluates model using corss-validation.')
    parser.add_argument('-m', '--model', required=True, help='keyword specifying the model that should be evaluated') 
    parser.add_argument('-d', '--data', required=True, help='keyword specifying the data set the model should be evaluated on.')
    parser.add_argument('-in', '--in_suffix', default='', help='suffix to input HP file name')

    args = parser.parse_args()

    print_results(args.model, args.data, args.in_suffix)
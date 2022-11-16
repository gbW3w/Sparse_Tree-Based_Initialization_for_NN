import torch
from time import process_time
from utils.metrics import *
from utils.utils_data import DataLoader_RepeatedStratifiedKFold

class Task():
    def __init__(self):
        self.object = object()

class TaskCollector():
    REGRESSION = Task()
    CLASSIF_BINARY = Task()
    CLASSIF_MULTICLASS = Task()

class Device():
    def __init__(self):
        self.object = object()

class DeviceCollector():
    CPU = Device()
    GPU = Device()

class Time():
    def __init__(self):
        self.start_called = False

    def start():
        raise NotImplementedError()

    def end():
        raise NotImplementedError()

    def check_startCalled(self):
        if not self.start_called:
            raise RuntimeError('Time: call Time.start() before Time.end().')

class TimeCUDA(Time):
    def __init__(self):
        super().__init__()
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        self.start_called = True
        self.start_event.record()

    def end(self):
        self.check_startCalled()
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)

class TimeCPU(Time):
    def __init__(self):
        super().__init__()

    def start(self):
        self.start_called = True
        self.start_time = process_time()

    def end(self):
        self.check_startCalled()
        return  process_time() - self.start_time

def get_timer(device):
    if device.type == 'cuda':
        return TimeCUDA()
    if device.type == 'cpu':
        return TimeCPU()
    raise ValueError()

def get_task(task: str, n_classes: int):
    # check argument validity
    valid_tasks = {'classification', 'regression'}
    if task not in valid_tasks:
        raise ValueError('task: must be one of %r' % valid_tasks)
    assert n_classes > 0, 'Number of classes must be positive'

    # set task
    if task == 'regression':
        task = TaskCollector.REGRESSION
    elif n_classes == 2:
        task = TaskCollector.CLASSIF_BINARY
    else:
        task = TaskCollector.CLASSIF_MULTICLASS
        assert n_classes > 1, 'Number of classes must be bigger than 1 for classification tasks.'

    return task

def get_metrics(DataLoader: DataLoader_RepeatedStratifiedKFold, method: str):
    logits = method_isMLP(method)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # set up metrics
    if DataLoader.task == 'regression':
        metrics = [MSELoss()]
    elif DataLoader.task == 'classification' and DataLoader.n_classes > 2:
        metrics = [CrossEntropyLoss(DataLoader.get_class_weights(), device, logits=logits), Accuracy(logits=logits)]
    elif DataLoader.task == 'classification' and DataLoader.n_classes == 2:
        metrics = [CrossEntropyLoss(DataLoader.get_class_weights(), device, logits=logits), Accuracy(logits=logits), AurocScoreBinary(logits=logits)]
    else:
        raise ValueError()
    metric_names = [metric.__name__ if hasattr(metric, '__name__') else type(metric).__name__ for metric in metrics]
    metrics = {name: metric for (name, metric) in zip(metric_names, metrics)}
    return metrics

def verify_method_dataName(method: str, data_name: str):
    # verify arguemnts
    valid_methods = ['RF', 'RF_init', 'XGB', 'XGB_init', 'DF', 'DF_init', 'MLP_rand_init', 'MLP_RF_init', 'MLP_XGB_init', 'MLP_DF_init', \
        'MLP_WT_init', 'MLP_LSUV_init', 'MLP_xavier_init', 'SAINT']
    if method not in valid_methods:
        raise ValueError(f'`method` not valid; must be one of {valid_methods}. Got {method}.')
    valid_data_names = ['housing', 'adult', 'bank', 'covertype', 'volkert', 'higgs', 'airbnb', 'heloc', 'blastchar', 'diamonds']
    if data_name not in valid_data_names:
        raise ValueError(f'`data_name` not valid; mus be one of {valid_data_names}. Got {data_name}.')

def method_isMLP(method):
    return method in ['MLP_rand_init', 'MLP_RF_init', 'MLP_XGB_init', 'MLP_DF_init', 'MLP_WT_init', 'MLP_LSUV_init', 'MLP_xavier_init', 'SAINT']
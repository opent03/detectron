import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle

def load_and_process_uci(pth=None):
    uci_dict = loadmat(pth if pth is not None else 'data/sample_data/uci_heart_processed.mat')
    x_train = uci_dict['train_data']
    x_test = uci_dict['ood_test_data']
    y_train = uci_dict['train_labels']
    y_test = uci_dict['ood_test_labels']
    x_train, y_train = shuffle(x_train, y_train.T)
    x_test, y_test = shuffle(x_test, y_test.T)
    return x_train, x_test, y_train, y_test
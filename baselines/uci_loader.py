import numpy as np
from scipy.io import loadmat
from sklearn.utils import shuffle
from sklearn.decomposition import PCA


def normalize(data, mean, std):
    return (data - mean) / std


def load_and_process_uci(n_components=9, pth=None):
    # load data --------------------------------------
    uci_dict = loadmat(pth if pth is not None else 'data/sample_data/uci_heart_processed.mat')
    x_train = uci_dict['train_data']
    x_test = uci_dict['ood_test_data']
    y_train = uci_dict['train_labels']
    y_test = uci_dict['ood_test_labels']
    x_train, y_train = shuffle(x_train, y_train.T)
    x_test, y_test = shuffle(x_test, y_test.T)
    # normalize --------------------------------------
    train_mean, train_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
    x_train = normalize(x_train, train_mean, train_std)
    x_test = normalize(x_test, train_mean, train_std)
    # PCA to realign ---------------------------------
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    x_train, x_test = pca.transform(x_train), pca.transform(x_test)
    return x_train, x_test, y_train, y_test
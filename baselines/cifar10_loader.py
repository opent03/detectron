import numpy as np
from sklearn.decomposition import PCA


def load_cifar10_512():
    with open('baselines/data/cifar10_x_train.npy', 'rb') as f:
        x_train = np.load(f)
    #with open('baselines/data/cifar10_x_test.npy', 'rb') as f:
    #    x_test = np.load(f)
    with open('baselines/data/cifar10_y_train.npy', 'rb') as f:
        y_train = np.load(f)
    #with open('baselines/data/cifar10_y_test.npy', 'rb') as f:
    #    y_test = np.load(f)
    return x_train, y_train

def load_cifar101_512():
    with open('baselines/data/cifar101_x_test.npy', 'rb') as f:
        x_test = np.load(f)
    with open('baselines/data/cifar10.1_v6_labels.npy', 'rb') as f:
        y_test = np.load(f)
    return x_test, y_test

def load_and_process_cifar(n_components=10):
    # load datasets -------------------------------
    x_train, y_train = load_cifar10_512()
    x_test, y_test = load_cifar101_512()
    
    xy_train = np.concatenate([x_train, np.expand_dims(y_train, axis=1)], axis=1)
    np.random.shuffle(xy_train)
    x_train, y_train = xy_train[:, :512], xy_train[:,512:]
    
    xy_test = np.concatenate([x_test, np.expand_dims(y_test, axis=1)], axis=1)
    np.random.shuffle(xy_test)
    x_test, y_test = xy_test[:, :512], xy_test[:,512:]
    # PCA -----------------------------------------
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    x_train_pca, x_test_pca = pca.transform(x_train), pca.transform(x_test)
    print('PCA explained variance ratio: ', sum(pca.explained_variance_ratio_))
    return x_train_pca, x_test_pca, y_train, y_test
    
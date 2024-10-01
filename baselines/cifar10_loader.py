import numpy as np
import sklearn
from sklearn.decomposition import PCA


def load_cifar10_512():
    with open('baselines/data/cifar10_x_train.npy', 'rb') as f:
        x_train = np.load(f)
    with open('baselines/data/cifar10_x_test.npy', 'rb') as f:
        x_test = np.load(f)
    with open('baselines/data/cifar10_y_train.npy', 'rb') as f:
        y_train = np.load(f)
    with open('baselines/data/cifar10_y_test.npy', 'rb') as f:
        y_test = np.load(f)
    return x_train, x_test, y_train, y_test

def load_cifar101_512():
    with open('baselines/data/cifar101_x_test.npy', 'rb') as f:
        x_test = np.load(f)
    with open('baselines/data/cifar10.1_v6_labels.npy', 'rb') as f:
        y_test = np.load(f)
    return x_test, y_test

def load_and_process_cifar(n_components=20, return_all=False):
    tmp = {}
    splits = ['train', 'val', 'test']
    kk = ['x', 'y']
    # load iid 
    for k in kk:
        for split in splits:
            with open('baselines/data/cifar10_' + k + '_' + split + '.npy', 'rb') as f:
                tmp[k + '_' + split] = np.load(f)
    
    with open('baselines/data/cifar101_x_test.npy', 'rb') as f:
                tmp['x_test_ood'] = np.load(f)
    with open('baselines/data/cifar10.1_v6_labels.npy', 'rb') as f:
                tmp['y_test_ood'] = np.load(f)
                
    tmp['x_test_ood'], tmp['y_test_ood'] = sklearn.utils.shuffle(tmp['x_test_ood'], tmp['y_test_ood'])
    #pca = PCA(n_components=n_components)
    #pca.fit(tmp['x_train'], tmp['y_train'])
    #print('PCA explained variance ratio: ', sum(pca.explained_variance_ratio_))
    #x_train, x_test = pca.transform(tmp['x_train']), pca.transform(tmp['x_test_ood'])
    if not return_all:
        return tmp['x_train'], tmp['x_test'], tmp['y_train'], tmp['y_test_ood']
    else:
        return tmp

'''
def load_and_process_cifar(n_components=20):
    # load datasets -------------------------------
    x_train, x_val, y_train, y_val = load_cifar10_512()
    x_test, y_test = load_cifar101_512()
    
    # shuffle train
    xy_train = np.concatenate([x_train, np.expand_dims(y_train, axis=1)], axis=1)
    np.random.shuffle(xy_train)
    x_train, y_train = xy_train[:, :512], xy_train[:,512:]
    
    # shuffle train
    xy_val = np.concatenate([x_val, np.expand_dims(y_val, axis=1)], axis=1)
    np.random.shuffle(xy_val)
    x_val, y_val = xy_val[:, :512], xy_val[:,512:]
    
    # shuffle test
    xy_test = np.concatenate([x_test, np.expand_dims(y_test, axis=1)], axis=1)
    np.random.shuffle(xy_test)
    x_test, y_test = xy_test[:, :512], xy_test[:,512:]
    
    # PCA -----------------------------------------
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    x_train_pca, x_test_pca = pca.transform(x_train), pca.transform(x_test)
    x_val_pca = pca.transform(x_val)
    print('PCA explained variance ratio: ', sum(pca.explained_variance_ratio_))
    return x_train_pca, x_val_pca, x_test_pca, y_train, y_val, y_test
    '''
    

import numpy as np
from sklearn.decomposition import PCA
import pickle

def load_and_process_camelyon17(n_components=20):
    with open('baselines/data/camelyon17_features.pkl', 'rb') as f:
        datadict = pickle.load(f)
    with open('baselines/data/camelyon17_labels.pkl', 'rb') as f:
        labelsdict = pickle.load(f)
    
    pca = PCA(n_components=n_components)
    pca.fit(datadict['x_train'])
    x_train, x_test = pca.transform(datadict['x_train']), pca.transform(datadict['x_test'])
    y_train, y_test = labelsdict['y_train'], labelsdict['y_test']
    return x_train, x_test, y_train, y_test
import numpy as np
from scipy import stats
from tqdm import tqdm

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


class Divergence:
    
    DIV = ['h', 'kl', 'js', 'skl']
    
    def __init__(self, name):
        if name not in Divergence.DIV:
            raise ValueError('divergence type not appropriate!')
        self.name = name
        self.bandwidth = 'scott'
    
    def _get_kde_estimates(self, bandwidth, data):
        data = data.T
        kernel = stats.gaussian_kde(data, bandwidth)
        return kernel.logpdf(data)
    
    def get_h_distance(self, X, Y):
        '''
        Returns the empirical estimate of the H-Min divergence between samples X and Y
        '''
        XY = np.concatenate([X, Y], axis=0)
        logprob_1 = self._get_kde_estimates(self.bandwidth, X)
        logprob_2 = self._get_kde_estimates(self.bandwidth, Y)
        logprob_m = self._get_kde_estimates(self.bandwidth, XY)
        
        vdiv = np.mean(-logprob_m) - min(np.mean(-logprob_1), np.mean(-logprob_2))
        
        return vdiv
    
    def get_kl_distance(self, X, Y, epsilon=1e-6):
        '''
        Given two datasets, fit 2 gaussians on each of them.
        Then, compute the kl-divergence between them depending on the mode.
        '''
        ed = lambda x: np.expand_dims(x, axis=1)
        mean_X, mean_Y = ed(np.mean(X, axis=0)), ed(np.mean(Y, axis=0))
        #cov_X, cov_Y = np.cov(X.T), np.cov(Y.T)
        cov_X = np.cov(X.T) + epsilon * np.eye(X.shape[-1])
        cov_Y = np.cov(Y.T) + epsilon * np.eye(Y.shape[-1])
        
        d = X.shape[-1]
        det = np.linalg.det
        cov_Y_inv = np.linalg.inv(cov_Y)
        tmp = np.dot(mean_X.T - mean_Y.T, np.dot(cov_Y_inv, (mean_X - mean_Y)))
        KL = 0.5 * (np.log(det(cov_Y)/det(cov_X)) - d + tmp + np.trace(np.dot(cov_Y_inv, cov_X)))
        
        return np.ravel(KL)[0] 
    
    def get_slow_kl_distance(self, X, Y):
        X, Y = X.T, Y.T
        kernel_X = stats.gaussian_kde(X, self.bandwidth)
        kernel_Y = stats.gaussian_kde(Y, self.bandwidth)
        
        prob_p = kernel_X.evaluate(X)
        logprob_p = np.log(prob_p + 1e-10)  # Added small epsilon for numerical stability
        logprob_q = kernel_Y.logpdf(X)
        
        kl = np.mean((logprob_p - logprob_q))  # Ensuring proper weighting by P(x)
        return kl
    
    def get_js_distance(self, X, Y):
        # mixture distribution
        idx = np.random.choice(len(X), int(len(Y)/2), replace=False)
        idy = np.random.choice(len(Y), int(len(Y)/2), replace=False)
        M = np.concatenate([X[idx], Y[idy]], axis=0)
        js = 0.5 * (self.get_kl_distance(X, M) + self.get_kl_distance(Y, M))
        return js
    
    
    def get_distance(self, X, Y):
        if self.name == 'h':
            return self.get_h_distance(X, Y)
        elif self.name == 'kl':
            return self.get_kl_distance(X, Y)
        elif self.name == 'js':
            return self.get_js_distance(X, Y)
        elif self.name == 'skl':
            return self.get_slow_kl_distance(X, Y)
        else:
            return self.get_js_distance(X, Y)


def _bootstrap(X, Y, size, shuffle=True):
    idx = np.random.choice(len(X), size, replace=True)
    idy = np.random.choice(len(Y), size, replace=True)
    XuY = np.concatenate((X[idx], Y[idy]), axis=0)
    if shuffle:
        np.random.shuffle(XuY)
    return XuY[:size//2], XuY[size//2:]

def permutation_test(distance, est, X, Y, perms=500, alpha=5e-2, enable_tqdm=False, bootstrap_size=100):
    f = tqdm if enable_tqdm else lambda x: x 
    # balance the samples after computing their raw distance
    '''
    if len(X) > len(Y):
        idx = np.random.choice(len(X), len(Y), replace=False)
        X = X[idx]
    '''
    #XuY = np.concatenate([X, Y], axis=0)
    distr = []
    for i in f(range(perms)):
        #np.random.shuffle(XuY)
        #idx = np.random.choice(len(XuY), len(Y))
        #X_, Y_ = XuY[idx[:len(Y)//2]], XuY[idx[len(Y)//2:]]
        #X_, Y_ = XuY[:len(Y)], XuY[len(Y):]
        #print(X_.shape, Y_.shape)
        X_, Y_ = _bootstrap(X, Y, size=bootstrap_size)
        tmp = distance.get_distance(X_, Y_)
        distr.append(tmp)
        
        #print('Time elapsed: {}'.format(end - start ))
        
    q = np.quantile(distr, 1-alpha)
    print(int(est > q))
    return int(est > q), distr, est

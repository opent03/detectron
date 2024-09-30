import os
import sys
import pickle
import numpy as np
import argparse
from baselines.cifar10_loader import load_and_process_cifar
from baselines.uci_loader import load_and_process_uci
from baselines.camelyon17_loader import load_and_process_camelyon17, load_and_process_camelyon17_mmd
from baselines.divergence import permutation_test
from baselines.dkmmd import DK_MMD
from matplotlib import pyplot as plt
from tqdm import tqdm

loader_dict = {
    'cifar10': load_and_process_cifar,
    'uci': load_and_process_uci,
    'camelyon17': load_and_process_camelyon17_mmd,
}

class DataDistribution:
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)
    
    def sample(self, n):
        idx = np.random.choice(self.length, n, replace=False)
        return self.data[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='mmd')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--loader_args', type=str, default="{'n_components':20}")
    parser.add_argument('--n_runs', type=int, default=100)
    parser.add_argument('--n_perms', type=int, default=500)
    parser.add_argument('--test_size', type=int, default=50)
    parser.add_argument('--oracle', type=bool, default=False)
    # logistics arguments
    parser.add_argument('--slurm_job_id', type=str, default='NA')
    parser.add_argument('--userid', type=str, default='opent03')
    args = parser.parse_args()
    args.loader_args = eval(args.loader_args)
    
    # Printing ----------------------------------------
    
    print('Algorithm: DK-MMD\nDataset: {}\n'.format(args.dataset))
    
    # Loading -----------------------------------------
    
    x_train, x_test, y_train, y_test = loader_dict[args.dataset](**args.loader_args)
    
    # Same training regime as DK-MMD paper for cifar10)
    x_val, x_test = x_test[:1000], x_test[1000:]
    x_train = x_train[np.random.choice(len(x_train), 1000, replace=False)]
    
    mmd_config = {
        'n': 1000,  
        'X': x_train,
        'Y': x_val,
        'n_features': 20,
        'hidden_dim': 32,
        'lr': 5e-5,
        'train_epochs': 500,
        'P': None,
        'Q': None
    }
    distance = DK_MMD(**mmd_config)
    
    # Checkpointing cringe -----------------------------
    
    checkpoint_path = '/checkpoint/{}/{}'.format(args.userid, args.slurm_job_id)
    
    # Does checkpoint path exist
    if not os.path.exists(checkpoint_path):
        checkpoint_path = 'checkpoint/{}/{}'.format(args.userid, args.slurm_job_id)
        os.makedirs(checkpoint_path, exist_ok=True)
    
    # Is DK-MMD trained
    if not os.path.exists(os.path.join(checkpoint_path, 'mmd_weights.pth')):
        print('No DK-MMD weights saved. Training...')
        distance.train()
        distance.save(checkpoint_path, 'mmd_weights.pth')
    else:
        distance.load(os.path.join(checkpoint_path, 'mmd_weights.pth'))
    
    # Are there a list of flags for TPR computation
    if not os.path.exists(os.path.join(checkpoint_path, 'flag_list')): # running on home computer
        print('No checkpoint found, starting fresh instance.')
        flag_list = []
    else:
        with open(os.path.join(checkpoint_path, 'flag_list'), 'rb') as f:
            flag_list = pickle.load(f)
            print('Checkpointed run, current progress: {}/{}.'.format(len(flag_list), args.n_runs))    
    
    # Permutation tests ---------------------------------
    
    print('Running permutation tests for {}'.format(args.name))
    for run in tqdm(range(len(flag_list), args.n_runs)):
        flag, _, _ = permutation_test(distance, 
                                    X=x_train, 
                                    Y=x_test, 
                                    perms=args.n_perms,
                                    max_size=args.test_size)
        flag_list.append(flag)
        with open(os.path.join(checkpoint_path, 'flag_list'), 'wb') as f:
            pickle.dump(flag_list, f)
        
    tpr = sum(flag_list) / args.n_runs
    
    # Write results to file ---------------------------
    
    with open('baselines/tprs_{}/{}_{}.log'.format(args.dataset, args.name, args.test_size), 'a+') as f:
        f.write('{}\n'.format(tpr))
    

if __name__ == '__main__':
    main()
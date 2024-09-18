import os
import sys
import pickle
import numpy as np
import argparse
from baselines.cifar10_loader import load_and_process_cifar
from baselines.uci_loader import load_and_process_uci
from baselines.divergence import Divergence, permutation_test
from matplotlib import pyplot as plt
from tqdm import tqdm

loader_dict = {
    'cifar10': load_and_process_cifar,
    'uci': load_and_process_uci
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='kl')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--loader_args', type=dict, default={'n_components':10})
    parser.add_argument('--n_runs', type=int, default=100)
    parser.add_argument('--n_perms', type=int, default=500)
    parser.add_argument('--test_size', type=int, default=50)
    parser.add_argument('--slurm_job_id', type=str, default='NA')
    parser.add_argument('--userid', type=str, default='opent03')
    args = parser.parse_args()
    
    # Loading -----------------------------------------
    
    x_train, x_test, y_train, y_test = loader_dict[args.dataset](**args.loader_args)
    idx = np.random.choice(len(x_test), args.test_size, replace=False)
    x_test = x_test[idx]
    
    # Restore checkpoint if exists --------------------
    
    checkpoint_path = '/checkpoint/{}/{}'.format(args.userid, args.slurm_job_id)
    if not os.path.exists(os.path.join(checkpoint_path, 'flag_list')): # running on home computer
        print('No checkpoint found, starting fresh instance.')
        flag_list = []
    else:
        with open(os.path.join(checkpoint_path, 'flag_list'), 'rb') as f:
            flag_list = pickle.load(f)
            print('Checkpointed run, current progress: {}/{}.'.format(len(flag_list), args.n_runs))

    # Permutation tests -------------------------------
    
    print('Running permutation tests for {} divergence'.format(args.name))
    distance = Divergence(args.name)
    est = distance.get_distance(x_train, x_test)
    
    for run in tqdm(range(len(flag_list), args.n_runs)):
        flag, _, _ = permutation_test(distance, est, x_train, x_test, perms=args.n_perms)
        flag_list.append(flag)
        with open(os.path.join(checkpoint_path, 'flag_list'), 'wb') as f:
            pickle.dump(flag_list, f)
        
    tpr = sum(flag_list) / args.n_runs
    
    # Write results to file ---------------------------
    
    with open('baselines/tprs_{}/{}_{}.log'.format(args.dataset, args.name, args.test_size), 'a+') as f:
        f.write('{}\n'.format(tpr))
    

if __name__ == '__main__':
    main()
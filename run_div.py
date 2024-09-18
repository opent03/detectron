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
    parser.add_argument('--test_size', type=int, default=20)
    args = parser.parse_args()
    
    # Loading -----------------------------------------
    
    x_train, x_test, y_train, y_test = loader_dict[args.dataset](**args.loader_args)
    
    # Permutation tests -------------------------------
    
    print('Running permutation tests for {} divergence'.format(args.name))
    distance = Divergence(args.name)
    flag_count = 0
    for run in tqdm(range(args.n_runs)):
        flag, _, _ = permutation_test(distance, x_train, x_test, perms=args.n_perms)
        flag_count += flag
    tpr = flag_count / args.n_runs
    
    # Write results to file ---------------------------
    
    with open('baselines/training_logs/{}.log'.format(args.name), 'a') as f:
        f.write('{}\n'.format(tpr))
    
    










if __name__ == '__main__':
    main()
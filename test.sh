#!/bin/bash

python run_div.py --name=kl --loader_args='{"n_components":20}' --test_size=50 --write_tprs=False
python run_div.py --name=js --loader_args='{"n_components":20}' --test_size=50 --write_tprs=False
python run_div.py --name=h --loader_args='{"n_components":20}' --test_size=50 --write_tprs=False
python run_mmd.py --loader_args='{"n_components":20}' --test_size=50 --write_tprs=False
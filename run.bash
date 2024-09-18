#!/bin/bash

search_dir=slurm_scripts
for entry in "$search_dir"/*
do
	sbatch $entry	
done
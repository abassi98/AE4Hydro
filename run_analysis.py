"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import glob
import os
import pickle
import sys

import pandas as pd


# number of ensemble members

nSeeds = 4
firstSeed = 300

# user inputs
folder = sys.argv[1]
experiment = sys.argv[2] # the options are "lstm_ae", "stat_lstm" and "lstm".
encoded_features = int(sys.argv[3]) # dimension of the encoded space in autoencoder

gpu = -1

if "global" in experiment:
    nSplits = 1
else:
    nSplits = 5

# This loop will run the evaluation procedure for all splits of all ensembles
for seed in range(firstSeed, firstSeed + nSeeds):  # loop through randomized ensembles
    for split in range(nSplits):  # number of k-fold splits
        # get the correct run directory by reading the screen report
        if "global" in experiment:
            fname = f"reports/{folder}/{experiment}_nldas.{seed}.out"
            f = open(fname)
            lines = f.readlines()
            print(f"Working on seed: {seed} -- file: {fname}")
            run_dir = lines[33].split('MultiBasinHydro2/')[1].split('attributes')[0]
            run_command = f"python LSTM_AE_main.py --gpu={gpu} --run_dir={run_dir} evaluate"

        elif experiment == "lstm_ae":
            fname = f"reports/{folder}/{experiment}_nldas.{seed}.{split}.out"
            print(f"Working on seed: {seed} -- file: {fname}")
            f = open(fname)
            lines = f.readlines()
            split_file = lines[13].split(': ')[1][:-1]
            split_num = lines[12].split(' ')[1][:-1]
            run_dir = lines[33].split('MultiBasinHydro2/')[1].split('attributes')[0]
            run_command = f"python LSTM_AE_main.py --gpu={gpu} --run_dir={run_dir} --split={split_num} --split_file={split_file} evaluate" 
        else:
            fname = f"reports/{folder}/{experiment}_nldas.{seed}.{split}.out"
            print(f"Working on seed: {seed} -- file: {fname}")
            f = open(fname)
            lines = f.readlines()
            split_file = lines[14].split(': ')[1][:-1]
            split_num = lines[13].split(' ')[1][:-1]
            run_dir = lines[33].split('MultiBasinHydro2/')[1].split('attributes')[0]
            run_command = f"python LSTM_main.py --gpu={gpu} --run_dir={run_dir} --split={split_num} --split_file={split_file} evaluate" 
    
        
        os.system(run_command)
        
        
        # grab the test output file for this split
        #file_seed = run_dir.split('seed')[1][:-1]
        results_file = glob.glob(f"{run_dir}/*lstm*seed*.p")[0]
        #print(results_file)
        with open(results_file, 'rb') as f:
            partial_dict = pickle.load(f)

        if "ae" in experiment and encoded_features > 0:
            results_file_enc = glob.glob(f"{run_dir}/*encoded*seed*.p")[0]
            with open(results_file_enc, 'rb') as f:
                partial_dict_enc = pickle.load(f)
        
       
    
        # store in a dictionary for this seed
        if split > 0:
            seed_dict.update(partial_dict)
            if "ae" in experiment and encoded_features > 0:
                seed_dict_enc.update(partial_dict_enc)
        else:
            seed_dict = partial_dict
            if "ae" in experiment and encoded_features > 0:
                seed_dict_enc = partial_dict_enc

        # screen report
        print(seed, split, len(seed_dict))
        
        # --- end of split loop -----------------------------------------

    # create the ensemble dictionary
    for basin in seed_dict:
        seed_dict[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
        if "ae" in experiment and encoded_features > 0:
            seed_dict_enc[basin].rename(columns={ 0 : seed}, inplace=True)
            
       
    
    if seed == 300: 
        ens_dict = seed_dict
        if "ae" in experiment and encoded_features > 0:
            ens_dict_enc = seed_dict_enc
    else:
        for basin in seed_dict:
            ens_dict[basin] = pd.merge(
                ens_dict[basin],
                seed_dict[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True)
            if "ae" in experiment and encoded_features > 0:
                ens_dict_enc[basin] = pd.merge(
                    ens_dict_enc[basin],
                    seed_dict_enc[basin][seed],
                    how='outer',
                    left_index=True,
                    right_index=True)
           

# --- end of seed loop -----------------------------------------

# calculate ensemble mean
for basin in ens_dict:
    print(ens_dict[basin].keys())
    simdf = ens_dict[basin].filter(regex='qsim_')
    ensMean = simdf.mean(axis=1)
    ens_dict[basin].insert(0, 'qsim', ensMean)


# save the ensemble results as a pickle
fname = f"analysis/results_data/{experiment}_{encoded_features}.pkl"
with open(fname, 'wb') as f:
    pickle.dump(ens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# save the encoded features as a pickle
if "ae" in experiment and encoded_features > 0:
    fname = f"analysis/results_data/encoded_{experiment}_{encoded_features}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(ens_dict_enc, f, protocol=pickle.HIGHEST_PROTOCOL)
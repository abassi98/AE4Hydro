
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

# This loop will run the evaluation procedure for all splits of all ensembles
for seed in range(firstSeed, firstSeed + nSeeds):  # loop through randomized ensembles
    # get the correct run directory by reading the screen report
    fname = f"reports/{folder}/{experiment}_nldas_ext.{seed}.out"
    f = open(fname)
    lines = f.readlines()
    print(f"Working on seed: {seed} -- file: {fname}")
    # retrieve run_dir
    for line in lines:
        if line.startswith("Sucessfully stored basin attributes"):
            run_dir = line.split('in ')[1].split('attributes')[0]
    
    run_command = f"python main.py --gpu={gpu} --run_dir={run_dir} evaluate"

    # run command
    os.system(run_command)
    
    # grab the test output file for this split
    results_file = glob.glob(f"{run_dir}/results*seed*.p")[0]
    with open(results_file, 'rb') as f:
        partial_dict = pickle.load(f)

    if encoded_features > 0:
        results_file_enc = glob.glob(f"{run_dir}/encoded*seed*.p")[0]
        with open(results_file_enc, 'rb') as f:
            partial_dict_enc = pickle.load(f)
       
    seed_dict = partial_dict
    if encoded_features > 0:
        seed_dict_enc = partial_dict_enc

    # screen report
    print(seed, len(seed_dict))
    
    # --- end of split loop -----------------------------------------

    # create the ensemble dictionary
    for basin in seed_dict:
        seed_dict[basin].rename(columns={'qsim': f"qsim_{seed}"}, inplace=True)
        if encoded_features > 0:
            seed_dict_enc[basin].rename(columns={ 0 : seed}, inplace=True)
            
       
    if seed == firstSeed: 
        ens_dict = seed_dict
        if encoded_features > 0:
            ens_dict_enc = seed_dict_enc
    else:
        for basin in seed_dict:
            ens_dict[basin] = pd.merge(
                ens_dict[basin],
                seed_dict[basin][f"qsim_{seed}"],
                how='inner',
                left_index=True,
                right_index=True)
            if encoded_features > 0:
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
if encoded_features > 0:
    fname = f"analysis/results_data/encoded_{experiment}_{encoded_features}.pkl"
    with open(fname, 'wb') as f:
        pickle.dump(ens_dict_enc, f, protocol=pickle.HIGHEST_PROTOCOL)
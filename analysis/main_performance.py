"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import pickle
import sys

import numpy as np
import pandas as pd
from typing import Dict

from performance_functions import (baseflow_index, bias, flow_duration_curve,
                                   get_quant, high_flows, low_flows, nse, abs_nse, sqrt_nse, FHV,
                                   stdev_rat, log_stdev_rat, zero_freq, FLV, FMM, kge, bias_RR)

# file name of ensemble dictionary is a user input
experiment = sys.argv[1]
encoded_features = int(sys.argv[2])
initial_seed = 300
num_seeds = 4

# load ensemble file
fname = f"analysis/results_data/{experiment}_{encoded_features}.pkl"
with open(fname, 'rb') as f:
    ens_dict = pickle.load(f)


# calcualte performance measures for ensembles
def compute_performance(ens_dict : Dict, experiment : str) -> None:
    stats = []
    bdex = -1
    for basin in ens_dict:
        # calcualte ensemble mean performance metrics
        obs5, sim5 = get_quant(ens_dict[basin], 0.05)
        obs95, sim95 = get_quant(ens_dict[basin], 0.95)
        obs0, sim0 = zero_freq(ens_dict[basin])
        obsH, simH = high_flows(ens_dict[basin])
        obsL, simL = low_flows(ens_dict[basin])
        e_nse = nse(ens_dict[basin])
        e_bias = bias(ens_dict[basin])
        e_stdev_rat = stdev_rat(ens_dict[basin])
        e_log_stdev_rat = log_stdev_rat(ens_dict[basin])
        obsFDC, simFDC = flow_duration_curve(ens_dict[basin])
        obsBF, simBF = baseflow_index(ens_dict[basin])
        e_abs_nse = abs_nse(ens_dict[basin])
        e_sqrt_nse = sqrt_nse(ens_dict[basin])
        e_FHV = FHV(ens_dict[basin])
        e_FLV = FLV(ens_dict[basin])
        e_kge = kge(ens_dict[basin])
        e_fmm = FMM(ens_dict[basin])
        e_RR = bias_RR(ens_dict[basin])
        

        # add ensemble mean stats to globaldictionary
        stats.append({
            'basin': basin,
            'nse': e_nse,
            'bias': e_bias,
            'stdev': e_stdev_rat,
            'obs5': obs5,
            'sim5': sim5,
            'obs95': obs95,
            'sim95': sim95,
            'obs0': obs0,
            'sim0': sim0,
            'obsL': obsL,
            'simL': simL,
            'obsH': obsH,
            'simH': simH,
            'obsFDC': obsFDC,
            'simFDC': simFDC,
            'obsBF' : obsBF,
            'simBF' : simBF,
            'abs_nse' : e_abs_nse,
            'sqrt_nse' : e_sqrt_nse,
            'FHV' : e_FHV,
            'FLV' : e_FLV,
            'kge' : e_kge,
            'FMM' : e_fmm,
            'bias_RR' : e_RR,
            'log_stdev' : e_log_stdev_rat,
        })


        # print basin-specific stats
        bdex = bdex + 1
        print(f"{basin} ({bdex} of {len(ens_dict)}) --- NSE: {stats[bdex]['nse']} --- mNSE: {stats[bdex]['abs_nse']} --- FHV: {stats[bdex]['FHV']}")
       

    # save ensemble stats as a csv file
    stats = pd.DataFrame(stats,
                        columns=[
                            'basin', 'nse', 'bias', 'stdev', 'obs5', 'sim5', 'obs95', 'sim95', 'obs0',
                            'sim0', 'obsL', 'simL', 'obsH', 'simH', 'obsFDC', 'simFDC','obsBF', 'simBF', 'abs_nse', 'sqrt_nse', 'FHV', 'FLV', 'kge', 'FMM', 'bias_RR', 'log_stdev',
                        ])
    fname = f"analysis/stats/{experiment}_{encoded_features}.csv"
    stats.to_csv(fname)

    # print to screen
    print('Mean NSE: ', stats['nse'].mean())
    print('Median NSE: ', stats['nse'].median())
    print('Num Failures: ', np.sum((stats['nse'] < 0).values.ravel()))

    # compute seed specifi statistics
    for seed in range(initial_seed, initial_seed+num_seeds):
        stats_seed = []
        for basin in ens_dict:
            df_seed = ens_dict[basin][[f"qsim_{seed}", 'qobs']].rename(columns={f"qsim_{seed}": 'qsim'})

            # calcualte ensemble mean performance metrics
            obs5, sim5 = get_quant(df_seed, 0.05)
            obs95, sim95 = get_quant(df_seed, 0.95)
            obs0, sim0 = zero_freq(df_seed)
            obsH, simH = high_flows(df_seed)
            obsL, simL = low_flows(df_seed)
            e_nse = nse(df_seed)
            e_bias = bias(df_seed)
            e_stdev_rat = stdev_rat(df_seed)
            e_log_stdev_rat = log_stdev_rat(df_seed)
            obsFDC, simFDC = flow_duration_curve(df_seed)
            obsBF, simBF = baseflow_index(df_seed)
            e_abs_nse = abs_nse(df_seed)
            e_sqrt_nse = sqrt_nse(df_seed)
            e_FHV = FHV(df_seed)
            e_FLV = FLV(df_seed)
            e_kge = kge(df_seed)
            e_fmm = FMM(df_seed)
            e_RR = bias_RR(df_seed)
            

            # add ensemble mean stats to globaldictionary
            stats_seed.append({
                'basin': basin,
                'nse': e_nse,
                'bias': e_bias,
                'stdev': e_stdev_rat,
                'obs5': obs5,
                'sim5': sim5,
                'obs95': obs95,
                'sim95': sim95,
                'obs0': obs0,
                'sim0': sim0,
                'obsL': obsL,
                'simL': simL,
                'obsH': obsH,
                'simH': simH,
                'obsFDC': obsFDC,
                'simFDC': simFDC,
                'obsBF' : obsBF,
                'simBF' : simBF,
                'abs_nse' : e_abs_nse,
                'sqrt_nse' : e_sqrt_nse,
                'FHV' : e_FHV,
                'FLV' : e_FLV,
                'kge' : e_kge,
                'FMM' : e_fmm,
                'bias_RR' : e_RR,
                'log_stdev' : e_log_stdev_rat,
            })

        # save stats seed as a csv file
        stats_seed = pd.DataFrame(stats_seed,
                        columns=[
                            'basin', 'nse', 'bias', 'stdev', 'obs5', 'sim5', 'obs95', 'sim95', 'obs0',
                            'sim0', 'obsL', 'simL', 'obsH', 'simH', 'obsFDC', 'simFDC','obsBF', 'simBF', 'abs_nse', 'sqrt_nse', 'FHV', 'FLV', 'kge', 'FMM', 'bias_RR', 'log_stdev',
                        ])
        fname = f"analysis/stats/{experiment}_{encoded_features}_{seed}.csv"
        stats_seed.to_csv(fname)

compute_performance(ens_dict, experiment)
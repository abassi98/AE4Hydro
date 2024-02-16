"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G., "Prediction 
in Ungauged Basins with Long Short-Term Memory Networks". submitted to Water Resources Research 
(2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import numpy as np


def nse(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    SSerr = np.mean(np.square(df["qsim"][idex] - df["qobs"][idex]))
    SStot = np.mean(np.square(df["qobs"][idex] - np.mean(df["qobs"][idex])))
    return 1 - SSerr / SStot


def abs_nse(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    SSerr = np.mean(np.abs(df["qsim"][idex] - df["qobs"][idex]))
    SStot = np.mean(np.abs(df["qobs"][idex] - np.mean(df["qobs"][idex])))
    return 1 - SSerr / SStot

def sqrt_nse(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    SSerr = np.mean(np.square(np.sqrt(df["qsim"][idex])- np.sqrt(df["qobs"][idex])))
    SStot = np.mean(np.square(np.sqrt(df["qobs"][idex]) - np.mean(np.sqrt(df["qobs"][idex]))))
    return 1 - SSerr / SStot

def r_coeff(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    SStot = np.sum((df["qobs"][idex] - np.mean(df["qobs"][idex])) * (df["qsim"][idex] - np.mean(df["qsim"][idex])))
    SSobs = np.sqrt(np.sum(np.square(df["qobs"][idex] - np.mean(df["qobs"][idex]))))
    SSsim = np.sqrt(np.sum(np.square(df["qsim"][idex] - np.mean(df["qsim"][idex]))))
    
    return SStot / (SSobs * SSsim)

def kge(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    r = r_coeff(df)
    beta = df["qsim"][idex].mean() / df["qobs"][idex].mean()
    alpha = df["qsim"][idex].std() / df["qobs"][idex].std()
    
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)



def get_quant(df, quant):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].quantile(quant)
    obs = df["qobs"][idex].quantile(quant)
    return obs, sim


def bias(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].mean()
    obs = df["qobs"][idex].mean()
    return (sim - obs) / obs

def bias_RR(df):
    """
    Yilmaz et al., 2008
    """
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    Num = (df["qsim"][idex] - df["qobs"][idex]).sum()
    Den = df["qobs"][idex].sum()
    return Num / Den


def stdev_rat(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = df["qsim"][idex].std()
    obs = df["qobs"][idex].std()
    return sim / obs


def log_stdev_rat(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = np.log(df["qsim"][idex] + 1e-08).std()
    obs = np.log(df["qobs"][idex] + 1e-08).std()
    return sim / obs

def zero_freq(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = (df["qsim"][idex] == 0).astype(int).sum()
    obs = (df["qobs"][idex] == 0).astype(int).sum()
    return obs, sim


def flow_duration_curve(df):
    obs33, sim33 = get_quant(df, 0.33)
    obs66, sim66 = get_quant(df, 0.66)
    sim = (np.log(sim33) - np.log(sim66)) / (0.66 - 0.33)
    obs = (np.log(obs33) - np.log(obs66)) / (0.66 - 0.33)
    return obs, sim

def FHV(df, quant=0.02):
    """
    Flow duration curve high segment value, Yilmaz et al., 2008
    """
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    df_sorted = df.loc[idex,:].sort_values(by=["qobs", "qsim"], ascending=False)
    cutoff = int(len(df["qobs"].to_numpy())*quant)
    obs_high, sim_high = df_sorted["qobs"].to_numpy()[:cutoff], df_sorted["qsim"].to_numpy()[:cutoff]
            
    return (sim_high - obs_high).sum() / obs_high.sum()
 
def FLV(df, quant=0.9):
    """
    Log Flow duration curve low segment value, Yilmaz et al., 20008
    """
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    df_sorted = df.loc[idex,:].sort_values(by=["qobs", "qsim"], ascending=False)
    cutoff = int(len(df["qobs"].to_numpy())*quant)
    obs_low, sim_low = df_sorted["qobs"].to_numpy()[cutoff::], df_sorted["qsim"].to_numpy()[cutoff::]
            
    #Num = np.sum( np.log(sim_low) - np.log(sim_low[-1]) ) - np.sum( np.log(obs_low) - np.log(obs_low[-1]) )
    #Den = np.sum( np.log(obs_low) - np.log(obs_low[-1]) )

    return  (sim_low - obs_low).sum() / obs_low.sum()
 

def FMM(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    sim = np.log(df["qsim"][idex].mean())
    obs = np.log(df["qobs"][idex].mean())
    return (sim - obs) / obs

def baseflow_index(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsQ = df["qobs"][idex].values
    simQ = df["qsim"][idex].values
    nTimes = len(obsQ)

    obsQD = np.full(nTimes, np.nan)
    simQD = np.full(nTimes, np.nan)
    obsQD[0] = obsQ[0]
    simQD[0] = simQ[0]

    # Davide et al., 2022
    c = 0.925
    for t in range(1, nTimes):
        obsQD[t] = min(obsQ[t], c * obsQD[t - 1] + (1.0 - c) / 2.0 * (obsQ[t] + obsQ[t-1])) 
        simQD[t] = min(simQ[t], c * simQD[t - 1] + (1.0 - c) / 2.0 * (simQ[t] + simQ[t-1])) 

   
    obs_ind = np.sum(obsQD[1:]) / np.sum(obsQ[1:])
    sim_ind = np.sum(simQD[1:]) / np.sum(simQ[1:])


    return obs_ind, sim_ind


def high_flows(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsMedian = df["qobs"][idex].median()
    obsFreq = len(df["qobs"][idex].index[(df["qobs"][idex] >= 9 * obsMedian)].tolist())
    simMedian = df["qsim"][idex].median()
    simFreq = len(df["qsim"][idex].index[(df["qsim"][idex] >= 9 * simMedian)].tolist())
    return obsFreq, simFreq


def low_flows(df):
    idex = df["qsim"].index[(df["qsim"] >= 0) & (df["qobs"] >= 0)].tolist()
    obsMedian = df["qobs"][idex].median()
    obsFreq = len(df["qobs"][idex].index[(df["qobs"][idex] <= 0.2 * obsMedian)].tolist())
    simMedian = df["qsim"][idex].median()
    simFreq = len(df["qsim"][idex].index[(df["qsim"][idex] <= 0.2 * simMedian)].tolist())
    return obsFreq, simFreq

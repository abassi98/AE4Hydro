import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from src.datautils import load_attributes
from pathlib import PosixPath
from scipy.stats import pearsonr
from src.datautils import HYDRO_NAMES, CLIM_NAMES, LANDSCAPE_NAMES
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 20 
mpl.rcParams['ytick.labelsize'] = 20 

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--experiment',
        type=str,
        default="enca",)
   
    cfg = vars(parser.parse_args())
    return cfg

if __name__ == '__main__':

    # Load encoded features
    cfg = get_args()
    encoded_features_vec = [1,2,3,5,27]
    us_states = gpd.read_file('../data/usa-states-census-2014.shp')
    experiment = cfg["experiment"]

    stat_caam = pd.read_csv(f"stats/caam_0.csv", sep=",")
    stat_caam.index = [str(s).rjust(8,"0") for s in stat_caam.loc[:,"basin"]]
    basins = stat_caam.index
    num_basins = stat_caam.shape[0]


    # load attributes
    camels_root = PosixPath("../data/basin_dataset_public_v1p2/")
    keep = CLIM_NAMES+ HYDRO_NAMES + LANDSCAPE_NAMES +  ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("../data/attributes.db", stat_caam.index, keep_attributes=keep)
    # insert lat and lon in encoded attributes
    lon = df_S["gauge_lon"]
    lat = df_S["gauge_lat"]
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
   

    markers = ["^", "o", "v", "<", ">", "s"]
    i=0
    line = np.linspace(-1,1,1000)
    # smooth nse copmarison
    sigma = 5

    
    
    # statistics distributions
    df_NSE = pd.DataFrame()
    df_BIAS = pd.DataFrame()
    df_STDEV = pd.DataFrame()
    df_R = pd.DataFrame()

    nse_caam = stat_caam["nse"]
    df_NSE["CAAM"] = nse_caam
    df_BIAS["CAAM"] = stat_caam["bias_std"]
    df_STDEV["CAAM"] = stat_caam["stdev"]
    df_R["CAAM"] = stat_caam["r_coeff"]
    
    # retrieve NSE
    for i in range(len(encoded_features_vec)):
        # plot nse on the CONUS
        stats = pd.read_csv(f"stats/{experiment}_{encoded_features_vec[i]}.csv", sep=",")
        stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
        nse = stats["nse"]

        ### retrieve NSE Distribution
        df_NSE[f"ENCA-{encoded_features_vec[i]}"] = stats["nse"]
        df_BIAS[f"ENCA-{encoded_features_vec[i]}"] = stats["bias_std"]
        df_STDEV[f"ENCA-{encoded_features_vec[i]}"] = stats["stdev"]
        df_R[f"ENCA-{encoded_features_vec[i]}"] = stats["r_coeff"]
    
    mpl.rcParams['xtick.labelsize'] = 10 
    mpl.rcParams['ytick.labelsize'] = 10 

    fig, ax = plt.subplots(1,1, figsize=(6,6), constrained_layout=True)
    colors = ["navy", "darkgreen",  "red", "chocolate", "darkorange", "orange", "goldenrod" ]
    ticks = np.arange(len(df_NSE.columns))
    labels = df_NSE.columns

    # NSE
    ax.grid()
    g = sns.boxplot(df_NSE,ax=ax, whis=(5,95), color="lightblue")
    ax.set_xlabel(" ")
    ax.axvline(0.5, color="black")
    ax.set_xticks(ticks, labels, fontsize=20, rotation=90 )
    ax.set_ylabel("NSE", rotation=90, fontweight="bold", fontsize=25)
    ax.set_ylim([-0.2,1])
    fig.savefig("figures/boxplot_NSE.png", dpi = 300)

    ### plot metrics (R, BIAS,LOG-STDEV)
    fig, ax = plt.subplots(1,3, figsize=(20,6))
    ticks = np.arange(len(df_NSE.columns))
    labels = df_NSE.columns
    # NSE
    ax[0].grid()
    g = sns.boxplot(df_R,ax=ax[0], whis=(5,95), color="lightblue")
    ax[0].set_xlabel(" ")
    ax[0].axvline(0.5, color="black")
    ax[0].set_xticks(ticks, labels, fontsize=20, rotation=90 )
    ax[0].set_ylabel("R", rotation=90, fontweight="bold", fontsize=25)
    ax[0].set_ylim([0.2,1])
    # BIAS
    ax[1].grid()
    g = sns.boxplot(df_BIAS, ax=ax[1], whis=(5,95), color="lightblue")
    ax[1].set_xlabel(" ")
    ax[1].axvline(0.5, color="black")
    ax[1].set_xticks(ticks, labels, fontsize=20,  rotation=90)
    ax[1].set_ylabel("BIAS", rotation=90, fontweight="bold", fontsize=25)
    ax[1].set_ylim([-0.6,0.6])
    # STDEV
    ax[2].grid()
    g = sns.boxplot(df_STDEV, ax=ax[2], whis=(5,95), color="lightblue")
    ax[2].set_xlabel(" ")
    ax[2].set_xticks(ticks, labels, fontsize=20, rotation=90)
    ax[2].axvline(0.5, color="black")
    ax[2].set_ylabel("STDEV", rotation=90, fontweight="bold", fontsize=25)
    ax[2].set_ylim([0,1.5])
    fig.tight_layout()
    fig.savefig("figures/boxplots.png", dpi = 300)

    # print latex table
    df_NSE_des = df_NSE.describe(percentiles=[.05,.25, .5, .75, .95]).round(2).loc[["mean", "min", "5%", "25%", "50%", "75%","95%", "max"],:].rename({"mean":" & Mean", "min":"& Min","5%":"& Q5", "25%":"& Q25", "50%":"\\textbf{NSE} & Median", "75%":"& Q75", "95%":"& Q95", "max":"& Max"}, axis=0)
    df_BIAS_des = df_BIAS.describe(percentiles=[.05,.25, .5, .75, .95]).round(2).loc[["mean", "min", "5%", "25%", "50%", "75%","95%", "max"],:].rename( {"mean":" & Mean", "min":"& Min","5%":"& Q5", "25%":"& Q25", "50%":"\\textbf{BIAS} & Median", "75%":"& Q75", "95%":"& Q95", "max":"& Max"}, axis=0)
    df_STDEV_des = df_STDEV.describe(percentiles=[.05,.25, .5, .75, .95]).round(2).loc[["mean", "min", "5%", "25%", "50%", "75%","95%", "max"],:].rename({"mean":"  & Mean", "min":"& Min","5%":"& Q5", "25%":"& Q25", "50%":"\\textbf{STDEV} & Median", "75%":"& Q75", "95%":"& Q95", "max":"& Max"}, axis=0)
    df_R_des = df_R.describe(percentiles=[.05,.25, .5, .75, .95]).round(2).loc[["mean", "min", "5%", "50%", "95%", "max"],:].rename({"mean":"  & Mean", "min":"& Min","5%":"& Q5", "25%":"& Q25", "50%":"\\textbf{R} & Median", "75%":"& Q75", "95%":"& Q95", "max":"& Max"}, axis=0)
    df = pd.concat((df_NSE_des, df_BIAS_des, df_STDEV_des), axis=0)
    latex_table = df.to_latex(escape=False)
    print(latex_table)

    # correalation table between NSE components
    names = df_NSE.columns
    index_names = ["NSE - R", "NSE - BIAS", "NSE - STDEV", "R - BIAS", "R - STDEV", "BIAS - STDEV"]
    l = []
    mask = df_NSE > 0.0
    fail_counts = len(basins) - mask.apply(lambda col: col.sum())
    print(fail_counts)
    
    l.append([df_NSE[mask][n].corr(df_R[mask][n]) for n in names])
    l.append([df_NSE[mask][n].corr(df_BIAS[mask][n]) for n in names])
    l.append([df_NSE[mask][n].corr(df_STDEV[mask][n]) for n in names])
    l.append([df_R[mask][n].corr(df_BIAS[mask][n]) for n in names])
    l.append([df_R[mask][n].corr(df_STDEV[mask][n]) for n in names])
    l.append([df_BIAS[mask][n].corr(df_STDEV[mask][n]) for n in names])
    corr_el = pd.DataFrame(l,columns=names, index=index_names).round(2)
    latex_table = corr_el.to_latex(escape=False)
    print(latex_table)

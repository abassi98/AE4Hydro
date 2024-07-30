import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from src.datautils import load_attributes, CLIM_NAMES, HYDRO_NAMES, LANDSCAPE_NAMES
from src.utils import clean_and_capitalize
from pathlib import PosixPath
from scipy.stats import pearsonr
from cmcrameri import cm
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
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    # Load encoded features
    cfg = get_args()
    
    us_states = gpd.read_file('data/usa-states-census-2014.shp')
    experiment = cfg["experiment"]

    # load caam and basins
    stat_caam = pd.read_csv(f"analysis/stats/caam_0.csv", sep=",")
    stat_caam.index = [str(s).rjust(8,"0") for s in stat_caam.loc[:,"basin"]]
    basins = stat_caam.index
    num_basins = stat_caam.shape[0]


    # load lat and lon
    camels_root = PosixPath("data/basin_dataset_public_v1p2/")
    keep = ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=keep)
    # insert lat and lon in encoded attributes
    lon = df_S["gauge_lon"]
    lat = df_S["gauge_lat"]

    markers = ["^", "o", "v", "<", ">", "s"]
    i=0
    line = np.linspace(-1,1,1000)
    
    
    # statistics distributions
    nse_caam= stat_caam["nse"]

    ### plot nse ENCA vs PREVIOUS ENCA in encoded_features_vec
    encoded_features_vec = [1,2,3,5, 27]
    # plot scatter delta nse vs attributes
    names_to_keep = ["baseflow_index", "aridity", "high_q_freq", "q95", "carbonate_rocks_frac", "elev_mean",
                         "frac_snow", "frac_forest", "soil_conductivity" ]
        
    fig,ax = plt.subplots(2,4, figsize=(30,10),gridspec_kw={'width_ratios': [2, 1, 2, 1]})
    for i in range(1,len(encoded_features_vec)): # i = 1,2,3,4 column = 0, 2, 0, 2
        row, column = (i-1) // 2, 2 * ((i-1 ) % 2 )
        print(row, column)
        ax[row, column].set_xlim(-128, -65)
        ax[row, column].set_ylim(24, 50)
        ax[row, column].set_axis_off()
        us_states.boundary.plot(color="black", ax=ax[row, column], linewidth=0.5)

        # retrieve stats
        stats = pd.read_csv(f"analysis/stats/{experiment}_{encoded_features_vec[i]}.csv", sep=",")
        stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
        nse = stats["nse"]

        # retrieve previous stats
        stats_prev = pd.read_csv(f"analysis/stats/{experiment}_{encoded_features_vec[i-1]}.csv", sep=",")
        stats_prev.index = [str(s).rjust(8,"0") for s in stats_prev.loc[:,"basin"]]
        nse_prev = stats_prev["nse"]

        color = np.clip(nse - nse_prev, -1,1)
        im = ax[row, column].scatter(x=lon, y=lat,c=color, s=100, cmap=cm.roma_r, vmin=-1,vmax=1 )
        ax[row, column].set_title(f"Delta NSE: ENCA-{encoded_features_vec[i]} and ENCA-{encoded_features_vec[i-1]}", fontsize=35, y=-0.2)
        ax[row, column+1].set_ylabel(f"NSE ENCA-{encoded_features_vec[i-1]}", fontsize=35)
        ax[row, column+1].set_xlabel(f"NSE ENCA-{encoded_features_vec[i]}", fontsize=35)
        ax[row, column+1].plot(line, line, c="black",ls="--")
        sc = ax[row, column+1].scatter(np.clip(nse, -1,1), np.clip(nse_prev,-1,1), c=color, s=100, cmap=cm.roma_r, vmin=-1,vmax=1 )
        cb = fig.colorbar(sc)
        cb.set_label("Delta NSE clipped in [-1,1]", fontsize=25) 
        ax[row, column+1].text(-0.8,0.8, f"r = {np.round(pearsonr(nse, nse_prev)[0], 2)}", fontsize=30)
        ax[row, column+1].grid()
        
   
    fig.tight_layout()
    fig.savefig(f"analysis/figures/plot_delta_nse_vs_prev_{experiment}.png", dpi = 300)
    plt.close()
    
    
    ### plot nse ENCA vs CAAM
    encoded_features_vec = [2,3]
    fig,ax = plt.subplots(1,4, figsize=(30,5),gridspec_kw={'width_ratios': [2, 1,2,1]})
    for i in range(len(encoded_features_vec)):   
        ind = i*2 
        # retrieve stat
        stats = pd.read_csv(f"analysis/stats/{experiment}_{encoded_features_vec[i]}.csv", sep=",")
        stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
        nse = stats["nse"]
        
        # plot US map
        ax[ind].set_xlim(-128, -65)
        ax[ind].set_ylim(24, 50)
        ax[ind].set_axis_off()
        us_states.boundary.plot(color="black", ax=ax[ind], linewidth=0.5)
    
        color = np.clip(nse - nse_caam, -1,1)
        im = ax[ind].scatter(x=lon, y=lat,c=color, s=100, cmap=cm.roma_r, vmin=-1,vmax=1 )
        ax[ind].set_title(f"Delta NSE: ENCA-{encoded_features_vec[i]} and CAAM", fontsize=35, y=-0.2)
        ax[ind+1].set_ylabel("NSE CAAM", fontsize=35)
        ax[ind+1].set_xlabel(f"NSE ENCA-{encoded_features_vec[i]}", fontsize=35)
        ax[ind+1].plot(line, line, c="black",ls="--")
        sc = ax[ind+1].scatter(np.clip(nse, -1,1), np.clip(nse_caam,-1,1), c=color, s=100, cmap=cm.roma_r, vmin=-1,vmax=1 )
        cb = fig.colorbar(sc)
        cb.set_label("Delta NSE clipped in [-1,1]", fontsize=25) 
        ax[ind+1].text(-0.8,0.8, f"r = {np.round(pearsonr(nse, nse_caam)[0], 2)}", fontsize=30)
        ax[ind+1].grid()
        fig.tight_layout()
        fig.savefig(f"analysis/figures/plot_delta_nse_vs_caam_{experiment}.png", dpi = 300)
       
        
   
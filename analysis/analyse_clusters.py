import glob
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from src.datautils import load_attributes, INVALID_ATTR_PLOT
import seaborn as sns
from src.utils import get_basin_list
from pathlib import PosixPath
from src.utils import clean_and_capitalize
from dadapy import data

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--run',
        type=int,
        default=1,)
    parser.add_argument(
        '--seed',
        type=int,
        default=4,)
    parser.add_argument(
        '--encoded_features',
        type=int,
        default=3,)
    

    cfg = vars(parser.parse_args())
    return cfg


if __name__ == '__main__':
    # get args
    cfg = get_args()
    run = cfg["run"]
    seed = cfg["seed"]
    encoded_features = cfg["encoded_features"]
    
    ### Load attributes
    camels_root = PosixPath("data/basin_dataset_public_v1p2/")
    #add_camels_attributes(camels_root)
    basins = get_basin_list()
    
    # load hydrological signatures
    df_hydro = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_hydro.txt", sep=";", index_col='gauge_id')
    df_hydro.index = [str(b).rjust(8,"0") for b in df_hydro.index]
    drop_basins = [b for b in df_hydro.index if b not in basins]
    df_hydro = df_hydro.drop(drop_basins, axis=0)
    runoff_ratio = df_hydro["runoff_ratio"]
    stream_elas = df_hydro["stream_elas"]
    df_hydro = df_hydro.drop(["runoff_ratio", "stream_elas"], axis=1)
   
    
    # load climatique attributes 
    df_clim = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_clim.txt", sep=";", index_col='gauge_id')
    df_clim.index = [str(b).rjust(8,"0") for b in df_clim.index]
    drop_basins = [b for b in df_clim.index if b not in basins]
    df_clim = df_clim.drop(drop_basins, axis=0)
    df_clim = df_clim.drop(["low_prec_timing"], axis=1)
    df_clim["runoff_ratio"] = runoff_ratio
    df_clim["stream_elas"] = stream_elas
    
    
    # load topological attributes
    df_topo = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_topo.txt", sep=";", index_col='gauge_id')
    df_topo.index = [str(b).rjust(8,"0") for b in df_topo.index]
    drop_basins = [b for b in df_topo.index if b not in basins]
    df_topo = df_topo.drop(drop_basins, axis=0)
    lon = df_topo["gauge_lon"]
    lat = df_topo["gauge_lat"]
   

    # load geological attributes
    df_geol = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_geol.txt", sep=";", index_col='gauge_id')
    df_geol.index = [str(b).rjust(8,"0") for b in df_geol.index]
    drop_basins = [b for b in df_geol.index if b not in basins]
    df_geol = df_geol.drop(drop_basins, axis=0)
   

    # load soil attributes
    df_soil = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_soil.txt", sep=";", index_col='gauge_id')
    df_soil.index = [str(b).rjust(8,"0") for b in df_soil.index]
    drop_basins = [b for b in df_soil.index if b not in basins]
    df_soil = df_soil.drop(drop_basins, axis=0)
  

    # load vegetation attributes
    df_vege = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_vege.txt", sep=";", index_col='gauge_id')
    df_vege.index = [str(b).rjust(8,"0") for b in df_vege.index]
    drop_basins = [b for b in df_vege.index if b not in basins]
    df_vege = df_vege.drop(drop_basins, axis=0)
   

    # create static attributes dataframe
    df_S = pd.concat([df_clim, df_hydro, df_topo, df_geol, df_soil, df_vege], axis=1)

    # get rid of non considered variables
    drop_names = [c for c in df_S.columns if c in INVALID_ATTR_PLOT]
    df_S = df_S.drop(drop_names, axis=1)
    df_S.insert(0, "Cluster", np.zeros(568), True)
    
    # Load the US states shapefile from GeoPandas datasets
    us_states = gpd.read_file('analysis/usa-states-census-2014.shp')
    # plot CONUS
    fig,ax = plt.subplots(1,1, figsize=(10,3))
    us_states.boundary.plot(color="black", ax=ax, linewidth=0.5)

    # retrieve number of clusters
    n_clust = len(glob.glob1(f"hidalgo/run{run}/","*.txt"))
    
    # retrieve encoded features and normalize
    enc = pd.read_csv(f"encoded_global_lstm_ae_{encoded_features}_{seed}.txt", sep=" ")
    enc.index = [str(b).rjust(8, "0") for b in enc.iloc[:,0]]
    enc = enc.iloc[:,1:]
    enc = (enc - enc.min()) / (enc.max()-enc.min()) # normaliaztion

    clusters = {}
    # load and plot clusters
    for c_id in range(n_clust):
        # load basins in the cluster
        c = pd.read_csv(f"hidalgo/run{run}/c{c_id+1}_enca_{encoded_features}_{seed}.txt", sep=",").loc[:,"x"]
        c = [str(b).rjust(8, "0") for b in c]
        df_S.loc[c, "Cluster"] = c_id
        # compute id with k star estimator
        X = enc.loc[c,:].to_numpy()
        d = data.Data(X)
        ids, ids_err, kstars, log_likelihoods = d.return_ids_kstar_gride(initial_id=20, n_iter=10, Dthr=23.92812698, d0=0.001, d1=100, eps=1e-07)
        # plot cluster and info
        ax.scatter(x=lon.loc[c], y=lat.loc[c], s=4, label=f"{len(c)} basins - ID (kstar)= {np.round(ids[-1],2)}")    
    

        
    # davefig
    fig.legend()
    fig.tight_layout()
    fig.savefig(f"hidalgo/run{run}/plot_{encoded_features}_{seed}.png")

    ### compute correlations of static features with cluster index
    # normalize
    df_S = (df_S - df_S.min())/ (df_S.max()-df_S.min())
    columns = df_S.columns
    
    corr = np.array((df_S.corr("spearman")))[1:,:1]
    fig, axs = plt.subplots(1,1, figsize=(10,10))
    fig.subplots_adjust(left=.5, right=1.0) 
    #axs.text(-1.,5.0, "[", rotation="horizontal", fontsize=400, fontstretch=1)
    names = [ clean_and_capitalize( df_S.keys()[i]) for i in range(1, len(df_S.keys()))]
    g = sns.heatmap(corr, annot=False, yticklabels=names,cmap="coolwarm", ax=axs, vmin=-1, vmax=1 )#cbar_kws={'label': 'Absolute Spearman Correlation'})
    axs.hlines([0, 11, 22, 25, 28, 39, 46], xmin=-101, xmax=axs.get_xlim()[1], color="black", clip_on = False)
    fig.text(0.002, 0.83, "Climate", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.60, "Hydrological", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.504, "Topo.", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.446, "Geol.", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.3, "Soil", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.10, "Vegetation", fontsize=15, rotation="vertical")
    axs.figure.axes[-1].yaxis.label.set_size(10)
    axs.figure.axes[-1].tick_params(labelsize=10)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize=12)
    g.set_xlabel("Learnt Features Principal Components", fontsize=30)
    file_corr = f"hidalgo/run{run}/spearman_enca_{encoded_features}_seed{seed}.png"
    #axs.set_title("Spearman Correlation Matrix", fontsize)
    fig.tight_layout()
    fig.savefig(file_corr)
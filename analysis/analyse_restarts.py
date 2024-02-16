import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
from src.datautils import INVALID_ATTR, load_attributes, add_camels_attributes, INVALID_ATTR
from pathlib import Path, PosixPath
from dadapy import data
import umap
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

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
        default="global_lstm_ae",)
   

    cfg = vars(parser.parse_args())
    return cfg

def clean_and_capitalize(input_string):
    # Split the input string into words
    words = input_string.split("_")
    out = ""
    


    for w in words:
        if w=="gages2":
             w="Catchment"
        if w=="freq":
             w="Frequency"
        if w=="dur":
             w="Duration"
        if w=="elev":
             w="Elevation"
        if w=="prec" or w=="p":
            w="Prec"
        if w=="frac":
            w="Fraction"
        if w=="geol":
            w="Geological"
        w = w.capitalize()
        if w=="Nse":
             w = "NSE"
        if w=="Forest":
            w="of Forest"
        if w=="Snow":
            w="of Snow"
        if w=="Statsgo":
            w="(STATSGO)"
        if w=="Pelletier":
             w="(Pelletier)"
        if w=="Pet":
            w="PET"
        if w=="Elas":
            w="ELAS"
        if w=="Fdc":
             w="FDC"
        if w=="Gvf":
             w="GVF"
        if w=="Aridity":
             w="Aridity Index"
        if w=="Hfd":
             w="HFD"
        if w=="Lai":
             w="LAI"
        out += w
        out += " "

    return out
if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    # Load encoded features
    cfg = get_args()
    encoded_features_vec = [0,3,27]
    experiment = cfg["experiment"]
    initseed = 300
    nseeds = 4
    fig, ax = plt.subplots(2,2, sharey=True)
    # plot nse AE vs STAT colore with all the attributes
    for ind, encoded_features in enumerate(encoded_features_vec):
        i = ind % 2
        j = ind // 2
        df = pd.DataFrame()
        for seed in range(initseed, initseed+nseeds):
            # plot nse on the CONUS
            if encoded_features == 0:
                stats = pd.read_csv(f"analysis/stats/{experiment}_stat_{encoded_features}_{seed}.csv", sep=",")
            else:
                stats = pd.read_csv(f"analysis/stats/{experiment}_{encoded_features}_{seed}.csv", sep=",")
            stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
            nse = stats["nse"]
            df[seed] = nse

        df.index = np.arange(1,569)
        g = sns.heatmap(df.transpose().clip(-1,1), vmin=-1, vmax=1, cmap="YlOrRd", ax=ax[i,j], annot=False)
        ax[i,j].set_xticks([100,200,300,400,500])
        g.set_xticklabels(["100","200","300","400","500"], rotation = 0, fontsize=10)
        g.set_yticklabels(["1","2","3", "4"], rotation = 0, fontsize=10)
        if encoded_features==0:
            ax[i,j].set_title("CAAM", rotation=0,  fontsize=20)
        else:
            ax[i,j].set_title(f"ENCA-{encoded_features}", rotation=0,  fontsize=20)
        ax[1,j].set_xlabel("Basins", rotation=0,  fontsize=15)
        ax[i,0].set_ylabel("Random Seed", rotation=90, fontsize=15, x=-2)
    
    #cbar = fig.colorbar(g, ax=ax, orientation='vertical', shrink=0.6)
    fig.tight_layout()
    fig.savefig(f"analysis/figures/restarts.png")

    

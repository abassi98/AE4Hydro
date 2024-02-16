import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from typing import List
import seaborn as sns
from sklearn.manifold import TSNE
import pickle
from src.datautils import INVALID_ATTR, load_attributes, add_camels_attributes, INVALID_ATTR_PLOT, INVALID_ATTR
from src.utils import clean_and_capitalize, get_basin_list
from pathlib import Path, PosixPath
from dadapy import data
from src.utils import str2bool
from sklearn.decomposition import PCA

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--firstseed',
        type=int,
        default=300,)
    parser.add_argument(
        '--nseeds',
        type=int,
        default=4,)
    parser.add_argument(
        '--encoded_features',
        type=int,
        default=3,)
    parser.add_argument(
        '--with_pca',
        type=str2bool,
        default=False,)
    
    
    cfg = vars(parser.parse_args())
    return cfg


def compute_ids_scaling(X, range_max = 2048, N_min = 20):
    "instantiate data class"
    _data = data.Data(coordinates = X,
                     maxk = 100)
    "compute ids scaling gride"
    ids_gride, ids_err_gride, rs_gride = _data.return_id_scaling_gride(range_max=range_max)
    "compute ids with twoNN + decimation"
    ids_twoNN, ids_err_twoNN, rs_twoNN = _data.return_id_scaling_2NN(N_min = N_min)
    return ids_gride, ids_twoNN, rs_gride, rs_twoNN



if __name__ == '__main__':
    ##########################################################
    # Load encoded features of chosen LSTM-AE model
    ##########################################################
    # Load encoded features
    cfg = get_args()
    firstseed = cfg["firstseed"]
    nseeds = cfg["nseeds"]
    encoded_features = cfg["encoded_features"]
    experiment = "global_lstm_ae"
    file_id = f"{experiment}_{encoded_features}"
    file_enc = f"analysis/results_data/encoded_{file_id}.pkl"
    file_stat = f"analysis/stats/{file_id}.csv"
    stats = pd.read_csv(file_stat, sep=",")
    stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
    with open(file_enc, 'rb') as f:
            dict_enc = pickle.load(f)
    
   
    # load attributes
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
    print("Hydrological signatures: ", df_hydro.columns)
    
    # load climatique attributes 
    df_clim = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_clim.txt", sep=";", index_col='gauge_id')
    df_clim.index = [str(b).rjust(8,"0") for b in df_clim.index]
    drop_basins = [b for b in df_clim.index if b not in basins]
    df_clim = df_clim.drop(drop_basins, axis=0)
    df_clim = df_clim.drop(["low_prec_timing"], axis=1)
    df_clim["runoff_ratio"] = runoff_ratio
    df_clim["stream_elas"] = stream_elas
    print("Climatique attributes: ", df_clim.columns)
    
    # load topological attributes
    df_topo = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_topo.txt", sep=";", index_col='gauge_id')
    df_topo.index = [str(b).rjust(8,"0") for b in df_topo.index]
    drop_basins = [b for b in df_topo.index if b not in basins]
    df_topo = df_topo.drop(drop_basins, axis=0)
    lon = df_topo["gauge_lon"]
    lat = df_topo["gauge_lat"]
    print("Topological attributes: ", df_topo.columns)

    # load geological attributes
    df_geol = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_geol.txt", sep=";", index_col='gauge_id')
    df_geol.index = [str(b).rjust(8,"0") for b in df_geol.index]
    drop_basins = [b for b in df_geol.index if b not in basins]
    df_geol = df_geol.drop(drop_basins, axis=0)
    print("Geological attributes: ", df_geol.columns)

    # load soil attributes
    df_soil = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_soil.txt", sep=";", index_col='gauge_id')
    df_soil.index = [str(b).rjust(8,"0") for b in df_soil.index]
    drop_basins = [b for b in df_soil.index if b not in basins]
    df_soil = df_soil.drop(drop_basins, axis=0)
    print("Soil attributes: ", df_soil.columns)

    # load vegetation attributes
    df_vege = pd.read_csv("data/basin_dataset_public_v1p2/camels_attributes_v2.0/camels_vege.txt", sep=";", index_col='gauge_id')
    df_vege.index = [str(b).rjust(8,"0") for b in df_vege.index]
    drop_basins = [b for b in df_vege.index if b not in basins]
    df_vege = df_vege.drop(drop_basins, axis=0)
    print("Vegetation attributes: ", df_vege.columns)
 
    # Load the US states shapefile from GeoPandas datasets
    us_states = gpd.read_file('data/usa-states-census-2014.shp')

    # create static attributes dataframe
    df_S = pd.concat([df_clim, df_hydro, df_topo, df_geol, df_soil, df_vege], axis=1)

    # get rid of non considered variables
    drop_names = [c for c in df_S.columns if c in INVALID_ATTR_PLOT]
    df_S = df_S.drop(drop_names, axis=1)

    # normalisze
    df_S = (df_S - df_S.min())/(df_S.max() - df_S.min()) # normalized in (0,1)
   

    # plot first seed
    seed = firstseed
    x_ticks = [i for i in range(encoded_features)]

    
    if seed == firstseed:
        enc = np.zeros((0,encoded_features))
        for key in dict_enc:
            enc = np.concatenate((enc, np.expand_dims(dict_enc[key][seed],axis=0)), axis=0)
        # print stats
        print(f"Mean features: ", np.mean(enc, 0))
        print(f"Std features: ", np.std(enc, 0))
        #enc = np.array(umap.UMAP(n_components=encoded_features, n_neighbors=500).fit_transform(enc))
        if cfg["with_pca"]:
            pca = PCA(n_components=encoded_features, svd_solver='full')
            enc = pca.fit_transform(enc)
            print(f"Pca variance ratios of seed {seed}: {pca.explained_variance_ratio_}")
        columns = [f"enc_{i+1}_{seed}" for i in range(encoded_features)]
        basins = [b for b in dict_enc.keys()]
        df_E = pd.DataFrame(enc, index=basins, columns = columns)
        

        df_E = (df_E - df_E.min())/(df_E.max() - df_E.min()) # normalized in (0,1)
        df_E.to_csv(f"encoded_{experiment}_{encoded_features}_{seed}.txt", sep=" ")
        num_basins = df_E.shape[0]
        df_E.to_csv(f"encoded_{experiment}_{encoded_features}.txt", sep=" ")
       

        # fig,ax = plt.subplots(1,encoded_features, figsize=(10,3))
        # for i in range(1,encoded_features+1):
        #     ax[i-1].set_xlim(-128, -65)
        #     ax[i-1].set_ylim(24, 50)
        #     #ax[i-1].set_title(f"FEATURE {i}", fontsize=15, y=-0.2)
        #     ax[i-1].set_axis_off()
        
        #     us_states.boundary.plot(color="black", ax=ax[i-1], linewidth=0.5)
        #     im = ax[i-1].scatter(x=lon, y=lat,c=df_E[f"enc_{i}_{seed}"], cmap="YlOrRd", s=4, vmin=0, vmax=1)
            
        # # Colorbar
        # #fig.colorbar(im, ax=ax,orientation="horizontal")
        # #fig.tight_layout()
        # save_file = f"analysis/figures/plot_{experiment}_{encoded_features}_seed{seed}.png"
        # fig.tight_layout()
        # fig.savefig(save_file)
           
        # 3d plot
        if encoded_features==3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(enc[:,0], enc[:,1], enc[:,2], s=2)
            fig.savefig(f"analysis/figures/3Dplot_{experiment}_{encoded_features}_seed{seed}.png")

        # concat ES
        df_ES = pd.concat([df_E, df_S], axis=1)
        

        ### plot correalation matrix ES Spearman
        columns = df_ES.columns
        corr = np.array(np.abs(df_ES.corr("spearman")))[encoded_features:,:encoded_features]
        fig, axs = plt.subplots(1,1, figsize=(10,10))
        fig.subplots_adjust(left=.5, right=1.0) 
        xlabels = np.arange(1, encoded_features+1)
        #axs.text(-1.,5.0, "[", rotation="horizontal", fontsize=400, fontstretch=1)
        names = [ clean_and_capitalize( df_ES.keys()[i]) for i in range(encoded_features, len(df_ES.keys()))]
        g = sns.heatmap(corr, annot=False, xticklabels=xlabels, yticklabels=names,cmap="YlOrRd", ax=axs, )#cbar_kws={'label': 'Absolute Spearman Correlation'})
        axs.hlines([0, 11, 22, 25, 28, 39, 46], xmin=-11, xmax=axs.get_xlim()[1], color="black", clip_on = False)
        #axs.axvline(-1.01, ymin=0.0, ymax=1.0, color="black", clip_on = False)
        #axs.axvline(3.0, ymin=0.0, ymax=1.0, color="black", clip_on = False)
        fig.text(0.015, 0.5, " Known Catchment Attributes", fontsize=30, rotation="vertical")
        fig.text(0.015, 0.15, "Signatures", fontsize=30, rotation="vertical")
        axs.figure.axes[-1].yaxis.label.set_size(10)
        axs.figure.axes[-1].tick_params(labelsize=10)
        g.set_xticks(x_ticks)
        g.set_xticklabels(range(1,encoded_features+1), rotation = 0, fontsize=15)
        g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize=15)
        g.set_xlabel("Relevant Features", fontsize=30)
        #sns.heatmap(corr_noise, xticklabels=corr.columns, yticklabels=corr.columns, cmap="bwr", ax=axs[1])
        file_corr = f"analysis/figures/ES_spearman_{experiment}_ef{encoded_features}_seed{seed}.png"
        #axs.set_title("Spearman Correlation Matrix", fontsize)
        fig.tight_layout()
        fig.savefig(file_corr)
    
    # plot the rest of the seed
    x_ticks = [2+4*i for i in range(encoded_features)]
    for seed in range(firstseed+1,firstseed + nseeds):
        # convvert to dataframe
        enc = np.zeros((0,encoded_features))
        for key in dict_enc:
            enc = np.concatenate((enc, np.expand_dims(dict_enc[key][seed],axis=0)), axis=0)

        # print stats
        print(f"Mean features: ", np.mean(enc, 0))
        print(f"Std features: ", np.std(enc, 0))

        #enc = np.array(umap.UMAP(n_components=encoded_features, n_neighbors=500).fit_transform(enc))
        if cfg["with_pca"]:
            pca = PCA(n_components=encoded_features, svd_solver='full')
            enc = pca.fit_transform(enc)
            print(f"Pca variance ratios of seed {seed}: {pca.explained_variance_ratio_}")
        columns = [f"enc_{i+1}_{seed}" for i in range(encoded_features)]
        basins = [b for b in dict_enc.keys()]
        df_E_seed = pd.DataFrame(enc, index=basins, columns = columns)

        df_E_seed = (df_E_seed - df_E_seed.min())/(df_E_seed.max() - df_E_seed.min()) # normalized in (0,1)
        df_E_seed.to_csv(f"encoded_{experiment}_{encoded_features}_{seed}.txt", sep=" ")
        num_basins = df_E_seed.shape[0]
        
        
        # fig,ax = plt.subplots(1,encoded_features, figsize=(10,3))
        # for i in range(1,encoded_features+1):
        #     ax[i-1].set_xlim(-128, -65)
        #     ax[i-1].set_ylim(24, 50)
        #     ax[i-1].set_axis_off()
        
        #     us_states.boundary.plot(color="black", ax=ax[i-1], linewidth=0.5)
        #     im = ax[i-1].scatter(x=lon, y=lat,c=df_E_seed[f"enc_{i}_{seed}"], cmap="YlOrRd", s=4, vmin=0, vmax=1)
            
        # save_file = f"analysis/figures/plot_{experiment}_{encoded_features}_seed{seed}.png"
        # fig.tight_layout()
        # fig.savefig(save_file)
        
        # concat ES
        df_E = pd.concat([df_E, df_E_seed], axis=1)

    # CONCAT
    new_order = []
    for i in range(encoded_features):
        for seed in range(firstseed, firstseed+nseeds):
            new_order.append(f"enc_{i+1}_{seed}")

    df_E = df_E[new_order]
    df_ES = pd.concat([df_E, df_S], axis=1)    
   
    ### plot correalation matrix ES Spearman for last three seeds
    columns = df_ES.columns
    corr = np.array(np.abs(df_ES.corr("spearman")))[4*encoded_features:,:4*encoded_features]
    fig, axs = plt.subplots(1,1, figsize=(10,10))
    fig.subplots_adjust(left=.5, right=1.0) 
    xlabels = np.concatenate((np.arange(1, encoded_features+1),np.arange(1, encoded_features+1),np.arange(1, encoded_features+1),np.arange(1, encoded_features+1)),axis=0)
    names = []
    for i in range(4*encoded_features, len(df_ES.keys())):
        names.append(clean_and_capitalize(df_ES.keys()[i]))
       
    g = sns.heatmap(corr, annot=False, xticklabels=xlabels, yticklabels=names,cmap="YlOrRd", ax=axs, vmin=0, vmax=1)#cbar_kws={'label': 'Absolute Spearman Correlation'})
    axs.hlines([0, 11, 22, 25, 28, 39, 46], xmin=-101, xmax=axs.get_xlim()[1], color="black", clip_on = False)
    fig.text(0.002, 0.83, "Climate", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.60, "Hydrological", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.504, "Topo.", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.446, "Geol.", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.3, "Soil", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.10, "Vegetation", fontsize=15, rotation="vertical")
    axs.figure.axes[-1].yaxis.label.set_size(10)
    axs.figure.axes[-1].tick_params(labelsize=10)
    g.set_xticks(x_ticks)
    g.set_xticklabels(range(1,encoded_features+1), rotation = 0, fontsize=10)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize=12)
    g.set_xlabel("Relevant Features Principal Components", fontsize=30)
    file_corr = f"analysis/figures/ES_spearman_{experiment}_ef{encoded_features}_all_seeds.png"
    fig.tight_layout()
    fig.savefig(file_corr)
       
       
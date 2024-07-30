import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from ..src.datautils import load_attributes, CLIM_NAMES, HYDRO_NAMES, LANDSCAPE_NAMES
from ..src.utils import clean_and_capitalize, get_basin_list, str2bool
from dadapy import data
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
        default=True,)
    
    parser.add_argument(
        '--experiment',
        type=str,
        default="global_lstm_ae_clim_atrue",)
   
    
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
    experiment = cfg["experiment"]
    stats = pd.read_csv(f"analysis/stats/{experiment}_{encoded_features}.csv", sep=",")
    stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
    with open(f"analysis/results_data/encoded_{experiment}_{encoded_features}.pkl", 'rb') as f:
            dict_enc = pickle.load(f)
    
    basins = [b for b in dict_enc.keys()]
   
    # get basins
    basins = get_basin_list()
    # load attributes with right order
    keep = CLIM_NAMES+ HYDRO_NAMES + LANDSCAPE_NAMES +  ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=keep)
    lat = df_S["gauge_lat"]
    lon = df_S["gauge_lon"]
    df_S = df_S.drop(["gauge_lat", "gauge_lon"], axis=1)
    # Load the US states shapefile from GeoPandas datasets
    us_states = gpd.read_file('data/usa-states-census-2014.shp')

    
    # plot first seed
    seed = firstseed
    x_ticks = [i for i in range(encoded_features)]
       
    # retrieve enc and plot pc on map 
    height = encoded_features * 7.0 /3.0
    keep = encoded_features
    fig, axs = plt.subplots(keep, nseeds, constrained_layout=True, figsize=(20,height))
    x_ticks = [2+4*i for i in range(keep)]
    df_E = None
    for seed in range(firstseed,firstseed + nseeds):
        j = seed % 300
        # convvert to dataframe
        enc = np.zeros((0,encoded_features))
        for key in dict_enc:
            enc = np.concatenate((enc, np.expand_dims(dict_enc[key][seed],axis=0)), axis=0)

        # print stats
        print(f"Mean features: ", np.mean(enc, 0))
        print(f"Std features: ", np.std(enc, 0))
        columns = [f"enc_{i+1}_{seed}" for i in range(encoded_features)]
        df_E_seed = pd.DataFrame(enc, index=basins, columns = columns)
        df_E_seed.to_csv(f"encoded/encoded_{experiment}_{encoded_features}_{seed}.csv", sep=" ")
        #enc = np.array(umap.UMAP(n_components=encoded_features, n_neighbors=500).fit_transform(enc))
        if cfg["with_pca"]:
            pca = PCA(n_components=encoded_features, svd_solver='full')
            enc = pca.fit_transform(enc)
            print(f"Pca variance ratios of seed {seed}: {pca.explained_variance_ratio_}")
       
        df_E_seed = pd.DataFrame(enc, index=basins, columns = columns)
        num_basins = df_E_seed.shape[0]
        # Enforce consistent sign convention
        for i in range(1,keep+1):
            if df_E_seed[f"enc_{i}_{seed}"][0] < 0:
                df_E_seed[f"enc_{i}_{seed}"] = - df_E_seed[f"enc_{i}_{seed}"]
            
        df_E_seed = (df_E_seed - df_E_seed.min())/(df_E_seed.max() - df_E_seed.min()) # normalized in (0,1)
        
        # plot principal components on map
        for i in range(1,keep+1):
            if keep > 1:
                ax = axs[i-1, j]
            else:
                ax = axs[j]
            
            ax.set_xlim(-128, -65)
            ax.set_ylim(24, 50)
            #ax[i-1].set_title(f"FEATURE {i}", fontsize=15, y=-0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            us_states.boundary.plot(color="black", ax=ax, linewidth=0.5)
            im = ax.scatter(x=lon, y=lat,c=df_E_seed[f"enc_{i}_{seed}"], cmap="viridis",vmin=0, vmax=1, s=10)
            if j==0:
                ax.set_ylabel(f"PC {i}",  fontsize=25)
            if i==1:
                ax.set_title(f"Restart {j+1}", fontsize=25)

        # concat ES
        df_E = pd.concat([df_E, df_E_seed], axis=1)

     # Colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.025, pad=0.04)
    cbar.ax.tick_params(labelsize=15)
    fig.savefig(f"analysis/figures/pca_{experiment}_{encoded_features}.png", dpi=300)

    # CONCAT
    new_order = []
    for i in range(encoded_features):
        for seed in range(firstseed, firstseed+nseeds):
            new_order.append(f"enc_{i+1}_{seed}")

    df_E = df_E[new_order]
    df_ES = pd.concat([df_E, df_S], axis=1)    
    
    ### plot correalation matrix ES Spearman for all seeds
    columns = df_ES.columns
    corr = np.array(np.abs(df_ES.corr("spearman")))[4*encoded_features:,:4*encoded_features]
    names = []
    for n in df_S.columns:
        names.append(clean_and_capitalize(n))
    x_ticks = [0.5+i for i in range(keep)]
    corr = corr[:, :4*keep]
    corr = corr.reshape(corr.shape[0],keep, nseeds)
    mean_corr = np.mean(corr, axis=-1)
    std_corr = np.std(corr, axis=-1)
    annotations = np.empty_like(mean_corr, dtype=object)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            annotations[i, j] = f'{mean_corr[i, j]:.2f} ± {std_corr[i,j]:.2f}'

    fig, axs = plt.subplots(1,1, figsize=(15,10))
      
    xlabels =np.arange(1, keep+1)
    
    g = sns.heatmap(mean_corr, annot=annotations, fmt='',xticklabels=xlabels, yticklabels=names,cmap="viridis", ax=axs, vmin=0, vmax=1)#cbar_kws={'label': 'Absolute Spearman Correlation'})
    axs.hlines([0, 9, 21, 24, 26, 34, 39], xmin=-101, xmax=axs.get_xlim()[1], color="black", clip_on = False)
    vlines = [i for i in range(keep+1)]
    axs.vlines(vlines, ymin=axs.get_ylim()[0], ymax=axs.get_ylim()[1], color="black", clip_on = False)
    fig.text(0.002, 0.85, "Climate", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.60, "Hydrological", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.43, "Topo.", fontsize=15, rotation="vertical")
    fig.text(0.0002, 0.38, "Geo.", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.27, "Soil", fontsize=15, rotation="vertical")
    fig.text(0.002, 0.09, "Vege.", fontsize=15, rotation="vertical")
    axs.figure.axes[-1].yaxis.label.set_size(10)
    axs.figure.axes[-1].tick_params(labelsize=10)
    g.set_xticks(x_ticks)
    g.set_xticklabels(range(1,keep+1), rotation = 0, fontsize=15)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize=15)
    g.set_xlabel("Relevant Features Principal Components", fontsize=15)
    file_corr = f"analysis/figures/ES_spearman_{experiment}_ef{encoded_features}.png"
    fig.tight_layout()
    fig.savefig(file_corr, dpi=300)
    
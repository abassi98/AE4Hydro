import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from cmcrameri import cm
import seaborn as sns
from sklearn.decomposition import PCA
from src.datautils import load_attributes, CLIM_NAMES, HYDRO_NAMES, LANDSCAPE_NAMES
from src.utils import clean_and_capitalize, get_basin_list, str2bool
import matplotlib as mpl
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"


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
        default="enca",)
    
    cfg = vars(parser.parse_args())
    if cfg["nseeds"] < 1 or cfg["encoded_features"] < 1:
        parser.error("--nseeds and --encoded_features must be positive.")
    return cfg



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
    # Use pandas' compatibility reader for files created with older pandas.
    dict_enc = pd.read_pickle(
        f"analysis/results_data/encoded_{experiment}_{encoded_features}.pkl"
    )
    
    # Use the same basin order for features, attributes, and map coordinates.
    basins = get_basin_list()
    missing_basins = set(basins) - dict_enc.keys()
    if missing_basins:
        raise ValueError(f"Encoded features are missing basins: {sorted(missing_basins)}")
    seeds = range(firstseed, firstseed + nseeds)
    for basin in basins:
        missing_seeds = set(seeds) - set(dict_enc[basin].columns)
        if missing_seeds:
            raise ValueError(f"Basin {basin} is missing seeds: {sorted(missing_seeds)}")
        if len(dict_enc[basin]) != encoded_features:
            raise ValueError(f"Basin {basin} does not have {encoded_features} features.")
    Path("analysis/encoded").mkdir(parents=True, exist_ok=True)
    Path("analysis/figures").mkdir(parents=True, exist_ok=True)
    # load attributes with right order
    keep = CLIM_NAMES + HYDRO_NAMES + LANDSCAPE_NAMES +  ["gauge_lat", "gauge_lon"]
    df_S = load_attributes("data/attributes.db", basins, keep_attributes=keep).loc[basins]
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
    keep = encoded_features # keep=8 with encoded_features > 5
    fig, axs = plt.subplots(keep, nseeds, constrained_layout=True, figsize=(20,height), squeeze=False)
    x_ticks = [2+4*i for i in range(keep)]
    df_E = None
    for j, seed in enumerate(seeds):
        enc = np.stack([dict_enc[basin][seed].to_numpy() for basin in basins])
        if not np.isfinite(enc).all():
            raise ValueError(f"Encoded features for seed {seed} contain non-finite values.")

        # print stats
        print(f"Mean features: ", np.mean(enc, 0).round(2))
        print(f"Std features: ", np.std(enc, 0).round(2))
        columns = [f"enc_{i+1}_{seed}" for i in range(encoded_features)]
        df_E_seed = pd.DataFrame(enc, index=basins, columns = columns)
        df_E_seed.to_csv(f"analysis/encoded/encoded_{experiment}_{encoded_features}_{seed}.csv", sep=" ")
        #enc = np.array(umap.UMAP(n_components=encoded_features, n_neighbors=500).fit_transform(enc))
        if cfg["with_pca"]:
            pca = PCA(n_components=encoded_features, svd_solver='full')
            enc = pca.fit_transform(enc)
            print(f"Pca variance ratios of seed {seed}: {pca.explained_variance_ratio_}")
       
        df_E_seed = pd.DataFrame(enc, index=basins, columns = columns)
        num_basins = df_E_seed.shape[0]
        # Enforce consistent sign convention
        for i in range(1,keep+1):
            if df_E_seed[f"enc_{i}_{seed}"].iloc[0] < 0:
                df_E_seed[f"enc_{i}_{seed}"] = - df_E_seed[f"enc_{i}_{seed}"]
            
        # Constant features map to zero instead of producing NaNs.
        feature_range = df_E_seed.max() - df_E_seed.min()
        df_E_seed = (df_E_seed - df_E_seed.min()) / feature_range.replace(0, 1)
        
        # plot principal components on map
        for i in range(1,keep+1):
            ax = axs[i-1, j]
            
            ax.set_xlim(-128, -65)
            ax.set_ylim(24, 50)
            #ax[i-1].set_title(f"FEATURE {i}", fontsize=15, y=-0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            us_states.boundary.plot(color="black", ax=ax, linewidth=0.5)
            im = ax.scatter(x=lon, y=lat,c=df_E_seed[f"enc_{i}_{seed}"], cmap=cm.batlow,vmin=0, vmax=1, s=10)
            if j==0:
                label = "PC" if cfg["with_pca"] else "Feature"
                ax.set_ylabel(f"{label} {i}",  fontsize=25)
            if i==1:
                ax.set_title(f"Restart {j+1}", fontsize=25)

        # concat ES
        df_E = pd.concat([df_E, df_E_seed], axis=1)

     # Colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.025, pad=0.04)
    cbar.ax.tick_params(labelsize=15)
    fig.savefig(f"analysis/figures/pca_{experiment}_{encoded_features}.png", dpi=300)
    plt.close(fig)

    # CONCAT
    new_order = []
    for i in range(encoded_features):
        for seed in range(firstseed, firstseed+nseeds):
            new_order.append(f"enc_{i+1}_{seed}")

    df_E = df_E[new_order]
    df_ES = pd.concat([df_E, df_S], axis=1)    
    
    ### plot correalation matrix ES Spearman for all seeds
    columns = df_ES.columns
    corr = df_ES.corr("spearman").loc[df_S.columns, new_order].abs().to_numpy()
    names = []
    for n in df_S.columns:
        names.append(clean_and_capitalize(n))
    x_ticks = [0.5+i for i in range(keep)]
    corr = corr.reshape(corr.shape[0],keep, nseeds)
    mean_corr = np.mean(corr, axis=-1)
    std_corr = np.std(corr, axis=-1)
    annotations = np.empty_like(mean_corr, dtype=object)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            annotations[i, j] = f'{mean_corr[i, j]:.2f} ± {std_corr[i,j]:.2f}'

    fig, axs = plt.subplots(1,1, figsize=(15,10))
      
    xlabels =np.arange(1, keep+1)
    
    g = sns.heatmap(mean_corr, annot=annotations, fmt='',xticklabels=xlabels, yticklabels=names,cmap=cm.batlow, ax=axs, vmin=0, vmax=1)#cbar_kws={'label': 'Absolute Spearman Correlation'})
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
    xlabel = "Relevant Features Principal Components" if cfg["with_pca"] else "Encoded Features"
    g.set_xlabel(xlabel, fontsize=15)
    fig.tight_layout()
    fig.savefig(f"analysis/figures/ES_spearman_{experiment}_ef{encoded_features}.png", dpi=300)
    plt.close(fig)

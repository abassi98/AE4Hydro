import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import LocallyLinearEmbedding
import plotly.express as px

from pathlib import Path, PosixPath
from dadapy import data, MetricComparisons
import umap
from sklearn.decomposition import PCA
from src.utils import get_basin_list
from src.datautils import load_attributes
from src.utils import clean_and_capitalize

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--feature',
        type=str,
        default="aridity",
        choices=['soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity',
       'soil_conductivity', 'max_water_content', 'sand_frac', 'silt_frac',
       'clay_frac', 'p_mean', 'pet_mean', 'p_seasonality', 'frac_snow',
       'aridity', 'high_prec_freq', 'high_prec_dur', 'low_prec_freq',
       'low_prec_dur', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max',
       'gvf_diff', 'elev_mean', 'slope_mean', 'area_gages2',
       'carbonate_rocks_frac', 'geol_permeability', 'q95', 'high_q_dur',
       'slope_fdc', 'high_q_freq', 'baseflow_index', 'runoff_ratio', 'q_mean',
       'low_q_freq', 'stream_elas', 'low_q_dur', 'q5', 'hfd_mean', 'gauge_lat', 'gauge_lon'])
    
    cfg = vars(parser.parse_args())
    return cfg

def plot():
    cfg= get_args()

    basins = get_basin_list()
    df_A = load_attributes("data/attributes.db", basins, drop_lat_lon=True, keep_features=None)
    df_S = load_attributes("data/attributes.db", basins, drop_lat_lon=False, keep_features=True)
    #df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_AS = pd.concat([df_A, df_S], axis=1)
    df_A = (df_A - df_A.min())/ (df_A.max()-df_A.min())
    # # imbalance
    # X = df_A.to_numpy()
    # d = MetricComparisons(X, maxk = X.shape[0]-1)
    # best_sets, best_imbs, all_imbs = d.greedy_feature_selection_full(n_coords=27, n_best=10, k=1)

    # fig, ax = plt.subplots(1,1)
    # ax.plot(range(1,28), best_imbs[:,1], "o-")
    # ax.set_ylabel(r'$\Delta(Reduced \; Space \rightarrow Full \; Space) $', fontsize=15)
    # ax.set_xlabel("Number of Variables in Reduced Space",fontsize=15)
    # fig.savefig("imb_known_attr.png")

    # fig, ax = plt.subplots(1,1)
    # #np.savetxt(f"analysis/imb_tuples_SvsE_{experiment}_{encoded_features}.txt", np.array(best_tuples, dtype=int))
    # f = open("imb_tuples_known_attr.txt", "w")
    # subsets_to_plot = [0,1,2,3,4,5,6,7,10,15,20,26]
    # for i in subsets_to_plot:
    #     imb = best_imbs[i]
    #     ax.scatter(imb[1], imb[0], c=imb[1], label=str(i+1),cmap="YlOrRd", vmin=-0.25, vmax=np.max(best_imbs[:,1]))
    #     f.write(str(df_A.columns[best_sets[i]].to_numpy()))
    #     f.write("\n")
    # f.close()
    # ax.plot([0, 1], [0, 1], 'k--')
    # ax.set_xlabel(r'$\Delta(Reduced \; Space \rightarrow Full \; Space) $', fontsize=15)
    # ax.set_ylabel(r'$\Delta(Full \; Space \rightarrow Reduced \;  Space) $', fontsize=15)
    # fig.legend(loc='lower right', fontsize=10, bbox_to_anchor=(0.9, 0.1))
    # fig.savefig("imb_plane_known_attr.png")

    # ### plot correalation matrix ES Spearman for last three seeds
    # columns = df_AS.columns
    # corr = np.array(np.abs(df_AS.corr("spearman")))
    # fig, axs = plt.subplots(1,1, figsize=(10,10))
    # fig.subplots_adjust(left=.5, right=1.0) 
    # g = sns.heatmap(corr, annot=False, xticklabels=columns, yticklabels=columns,cmap="YlOrRd", ax=axs )#cbar_kws={'label': 'Absolute Spearman Correlation'})
    # axs.hlines([0.0, 27, 39], xmin=-101, xmax=axs.get_xlim()[1], color="black", clip_on = False)
    # axs.figure.axes[-1].yaxis.label.set_size(10)
    # axs.figure.axes[-1].tick_params(labelsize=10)
    # file_corr = "ES_spearman_attr+sig.png"
    # fig.tight_layout()
    # fig.savefig(file_corr)

    # compute gride star
    X = df_A.to_numpy()
  
    d = data.Data(X)
    # gride
    ids_gride, ids_err_gride, rs_gride = d.return_id_scaling_gride(range_max=X.shape[0]-1)
    # fig, ax = plt.subplots(1,1)
    # ax.plot(rs_gride, ids_gride, "o-")
    # fig.savefig("gride_known_attr.png")
    ids, ids_err, kstars, log_likelihoods = d.return_ids_kstar_gride(initial_id=10, n_iter=5, Dthr=23.92812698, d0=0.001, d1=100, eps=1e-06)
    
    # compute lle
    print(f"Kstar used: {np.min(kstars)}")
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=4, reg=0.001, eigen_solver='dense', tol=1e-06, max_iter=100, method='standard', neighbors_algorithm='auto', random_state=42, n_jobs=4)
    Y = lle.fit_transform(X)
   

    # plot 3d projections
    fig, ax = plt.subplots(2,2, subplot_kw={'projection': '3d'}, figsize=(8,8))

    c = [0,1,2,3]
    for i in range(2):
        for j in range(2):
            c.remove(i*2+j)
            sc = ax[i,j].scatter(Y[:,c[0]], Y[:,c[1]], Y[:,c[2]], s=5, c=df_AS[cfg["feature"]])
            c = [0,1,2,3]

    fig.suptitle(clean_and_capitalize(cfg["feature"]), fontsize=25)
    fig.tight_layout()
    fig.colorbar(sc, ax=ax, shrink=0.5, orientation="horizontal", location="bottom")
    plt.show()

if __name__=="__main__":
    plot()
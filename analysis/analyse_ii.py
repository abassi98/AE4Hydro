from dadapy import data
from pathlib import Path, PosixPath
from src.datasets import CamelDataset
import argparse
from src.utils import get_basin_list
from src.datautils import load_attributes
import pandas as pd
import matplotlib.pyplot as plt

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=300,)
    parser.add_argument(
        '--encoded_features',
        type=int,
        default=3,)
    parser.add_argument(
        '--experiment',
        type=str,
        default="global_lstm_ae",)
   

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
    args = get_args()
    # initialize test dataset
    basins = get_basin_list()
    ds_test = CamelDataset(
            camels_root=Path('data/basin_dataset_public_v1p2/'),
            dates = [pd.to_datetime('01101995', format='%d%m%Y'), pd.to_datetime('29092010', format='%d%m%Y')],
            basins=basins,
            is_train=True,
            db_path = "data/attributes.db",
            no_static=True)
    
    Y = pd.DataFrame(ds_test.y.squeeze().detach().numpy(), index=basins)

    df_E = pd.read_csv(f"encoded_global_lstm_ae_{args['encoded_features']}_{args['seed']}.txt", sep=" ")
    index = df_E.iloc[:,0]
    df_E = df_E.iloc[:,1:]
    index = [str(b).rjust(8,"0") for b in index]
    df_E.index = index
    df_A = load_attributes("data/attributes.db", basins, drop_lat_lon=False, keep_features=None)
    df_S = load_attributes("data/attributes.db", basins, drop_lat_lon=False, keep_features=True)
    df_A = df_A.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_A = (df_A - df_A.min())/(df_A.max() - df_A.min()) # normalized in (0,1)
    df_S = (df_S - df_S.min())/(df_S.max() - df_S.min()) # normalized in (0,1)
    
    df_AS = pd.concat([df_A, df_S], axis=1)
    df_SE = pd.concat([df_S, df_E], axis=1)
   
    ### imbalance Signatures against Encoded Features
    d = data.Data(df_S.to_numpy())
    d_target = data.Data(df_E.to_numpy())
    d_target.compute_distances()
    target_ranks = d_target.dist_indices
    best_sets, best_imbs, _ = d.greedy_feature_selection_target(target_ranks, n_coords=12, k=1, n_best=100, symm=False) 

    # plot
    fig, ax = plt.subplots(1,1)
    ax.plot(range(1,13), best_imbs[:,1], "o-", label="Sig -> Encoded")
    ax.plot(range(1,13), best_imbs[:,0], "o-", label="Encoded -> Sig")
    ax.set_ylabel(r'$\Delta(Attributes \; Space \rightarrow Runoff \; Space) $', fontsize=15)
    ax.set_xlabel("Number of Variables in Reduced Space",fontsize=15)
    fig.legend()
    fig.savefig(f"imb_sig_encoded_{args['encoded_features']}_{args['seed']}.png")   

    # print on file 
    f = open(f"imb_tuples_sig_encoded_{args['encoded_features']}_{args['seed']}.txt", "w")
    for i in range(best_imbs.shape[0]):
        imb = best_imbs[i]
        f.write(str(df_S.columns[best_sets[i]].to_numpy()))
        f.write("\n")

    f.close()
    
   
    # ### imbalance Attributes against Y
    # d = data.Data(df_A.to_numpy())
    # d_target = data.Data(Y.to_numpy())
    # d_target.compute_distances()
    # target_ranks = d_target.dist_indices
    # best_sets, best_imbs, _ = d.greedy_feature_selection_target(target_ranks, n_coords=27, k=1, n_best=100, symm=False) 

    # # plot
    # fig, ax = plt.subplots(1,1)
    # ax.plot(range(1,28), best_imbs[:,1], "o-", label="Attr -> Runoff")
    # ax.plot(range(1,28), best_imbs[:,0], "o-", label="Runoff -> Attr")
    # ax.set_ylabel(r'$\Delta(Attributes \; Space \rightarrow Runoff \; Space) $', fontsize=15)
    # ax.set_xlabel("Number of Variables in Reduced Space",fontsize=15)
    # fig.legend()
    # fig.savefig("imb_attributes_runoff.png")   

    # # print on file 
    # f = open("imb_tuples_attr_runoff.txt", "w")
    # for i in range(best_imbs.shape[0]):
    #     imb = best_imbs[i]
    #     f.write(str(df_A.columns[best_sets[i]].to_numpy()))
    #     f.write("\n")

    # f.close()

    # ### imbalance Signatures against Y
    # d = data.Data(df_S.to_numpy())
    # d_target = data.Data(Y.to_numpy())
    # d_target.compute_distances()
    # target_ranks = d_target.dist_indices
    # best_sets, best_imbs, _  = d.greedy_feature_selection_target(target_ranks, n_coords=12, k=1, n_best=100, symm=False) 

    # # plot
    # fig, ax = plt.subplots(1,1)
    # ax.plot(range(1,13), best_imbs[:,1], "o-", label="Sig -> Runoff")
    # ax.plot(range(1,13), best_imbs[:,0], "o-", label="Runoff -> Sig")
    # ax.set_ylabel(r'$\Delta(Signatures \; Space \rightarrow Runoff \; Space) $', fontsize=15)
    # ax.set_xlabel("Number of Variables in Reduced Space",fontsize=15)
    # fig.legend()
    # fig.savefig("imb_signature_runoff.png")   

    # # print on file 
    # f = open("imb_tuples_sign_runoff.txt", "w")
    # for i in range(best_imbs.shape[0]):
    #     imb = best_imbs[i]
    #     f.write(str(df_S.columns[best_sets[i]].to_numpy()))
    #     f.write("\n")

    # f.close()

  
    # ### imbalance Signatures + Attributes against Y
    # d = data.Data(df_AS.to_numpy())
    # d_target = data.Data(Y.to_numpy())
    # d_target.compute_distances()
    # target_ranks = d_target.dist_indices
    # best_sets, best_imbs, _  = d.greedy_feature_selection_target(target_ranks, n_coords=39, k=1, n_best=100, symm=False) 

    # # plot
    # fig, ax = plt.subplots(1,1)
    # ax.plot(range(1,40), best_imbs[:,1], "o-", label="Attr+Sig -> Runoff")
    # ax.plot(range(1,40), best_imbs[:,0], "o-", label="Runoff -> Attr + Sig")
    # ax.set_ylabel(r'$\Delta(Signatures \; Space \rightarrow Runoff \; Space) $', fontsize=15)
    # ax.set_xlabel("Number of Variables in Reduced Space",fontsize=15)
    # fig.legend()
    # fig.savefig("imb_sign_attr+sign_runoff.png")   

    # # print on file 
    # f = open("imb_tuples_attr+sign_runoff.txt", "w")
    # for i in range(best_imbs.shape[0]):
    #     imb = best_imbs[i]
    #     f.write(str(df_AS.columns[best_sets[i]].to_numpy()))
    #     f.write("\n")

    # f.close()

    ### imbalance Signatures + Encoded against Y
    d = data.Data(df_SE.to_numpy())

    d_target = data.Data(Y.to_numpy())
    d_target.compute_distances()
    target_ranks = d_target.dist_indices
    best_sets, best_imbs, _  = d.greedy_feature_selection_target(target_ranks, n_coords=15, k=1, n_best=100, symm=False) 

    # plot
    fig, ax = plt.subplots(1,1)
    ax.plot(range(1,16), best_imbs[:,1], "o-", label="Sig+Enc -> Runoff")
    ax.plot(range(1,16), best_imbs[:,0], "o-", label="Runoff -> Sig +  Enc")
    ax.set_ylabel(r'$\Delta(Signatures \; Space \rightarrow Runoff \; Space) $', fontsize=15)
    ax.set_xlabel("Number of Variables in Reduced Space",fontsize=15)
    fig.legend()
    fig.savefig(f"imb_sign+enc_runoff_{args['encoded_features']}_{args['seed']}.png")   

    # print on file 
    f = open(f"imb_tuples_sig+enc_runoff_{args['encoded_features']}_{args['seed']}.txt", "w")
    for i in range(best_imbs.shape[0]):
        imb = best_imbs[i]
        f.write(str(df_SE.columns[best_sets[i]].to_numpy()))
        f.write("\n")

    f.close()

  

import argparse
import pickle
import matplotlib.pyplot as plt
from src.utils import get_basin_list
import pandas as pd
import numpy as np

def get_args():
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--t_init',
        type=int,
        default=1000,
        help="Day in the validation period to start the plot.")
    
    parser.add_argument(
        '--t_delta',
        type=int,
        default=500,
        help="Number of days to plot in the hydrograph.")
   

    cfg = vars(parser.parse_args())
    return cfg


def evaluate():
    # get args
    cfg = get_args()
    t_init = cfg["t_init"]
    t_delta = cfg["t_delta"]

    # get basins
    #basins = get_basin_list()
    basins = ["05120500", "05123400", "06332515", "06339500", "06406000", "06447000", "06450500", "06452000", 
              "07148400", "07301500", "08324000", "09404450", "10258500", "10259200","13083000"]
    models = {}
    # retrieve caam
    with open('analysis/results_data/global_lstm_ae_stat_0.pkl', 'rb') as f:
        models["CAAM"] = pickle.load(f)
        
    # retrieve models
    encoded_features = [2,3,27]
    for e in encoded_features:
        with open(f'analysis/results_data/global_lstm_ae_{e}.pkl', 'rb') as f:
            models[f"ENCA-{e}"]  = pickle.load(f)

    # load basin names
    info = pd.read_csv("data/basin_dataset_public_v1p2/basin_metadata/gauge_information.txt",engine="python",sep="\t")
    names = info.iloc[:,2]
    index = info.iloc[:,1]
    index = [str(b).rjust(8,"0") for b in index]
    names_dict = {}
    for i in range(len(names)):
        if index[i] in basins:
            names_dict[index[i]] = names[i]

    keys = list(models.keys())
    # plot hydrographs
    for  i in range(len(models)):
        # load model, previous model and caam series
        caam = models["CAAM"]
        if i==0:
            model = models["CAAM"]
        else:
            prev = models[keys[i-1]]
            model = models[keys[i]]

        # def figures
        fig_s, ax_s = plt.subplots(5,3, figsize = (20,12), sharex=True)
        #fig_h, ax_h = plt.subplots(5,3)
        for ind, basin in enumerate(basins):
            ix, iy = ind // 3, ind % 3

            obs = caam[basin]["qobs"]
            qsim_model = model[basin]["qsim"]

            # plot time series of hydrograph
            ax_s[ix,iy].plot(obs.squeeze()[t_init:t_init+t_delta], label="Observed", color="dodgerblue")
            if i>0:
                qsim_caam = caam[basin]["qsim"]
                ax_s[ix, iy].plot(qsim_caam.squeeze()[t_init:t_init+t_delta], label=keys[0], color="sandybrown")
                if i>1:
                    qsim_prev = prev[basin]["qsim"]
                    ax_s[ix, iy].plot(qsim_prev.squeeze()[t_init:t_init+t_delta], label=keys[i-1], color="turquoise")
            
            ax_s[ix, iy].plot(qsim_model.squeeze()[t_init:t_init+t_delta], label=keys[i], color="limegreen")
            ax_s[ix, 0].set_ylabel("Streamflow (mm/day)", fontsize=15)
            ax_s[ix, iy].set_title(f"Catchment {basin}", fontsize=15)        
            
            # # plot histogram
            # fig, ax = plt.subplots(1,1)
            # ax.hist(np.log(obs.squeeze()+ 1e-08), label="Observed", histtype="step",bins=100,color="dodgerblue")
            # if i>0:
            #     qsim_caam = caam[basin]["qsim"]
            #     ax.hist(np.log(qsim_caam.squeeze() + 1e-08), label=keys[0], histtype="step",bins=100,  color="sandybrown")
            #     if i>1:
            #         qsim_prev = prev[basin]["qsim"]
            #         ax.hist(np.log(qsim_prev.squeeze() + 1e-08), label=keys[i-1], histtype="step",bins=100, color="turquoise")

            # ax.hist(np.log(qsim_model.squeeze()+ 1e-08), label=keys[i], histtype="step",bins=100,color="limegreen")
            # fig.legend(fontsize=10, loc="upper right")
            # ax.set_xlabel("Streamflow (mm/day)", fontsize=15)
            # ax.set_title(names_dict[basin], fontsize=15, loc="center")
            # fig.tight_layout()
            # fig.savefig(f"hydrographs/hist/{keys[i]}/basin_{basin}.png")        
            # plt.close()
        handles, labels = ax_s[4,2].get_legend_handles_labels()
        fig_s.legend(handles,labels, fontsize=10)
        #fig_s.legend(fontsize=10, bbox_to_anchor=[0.1, 0.9], loc="upper left")
        fig_s.tight_layout()
        fig_s.savefig(f"hydrographs/series/{keys[i]}/f09.png")
    

if __name__ == "__main__":
    evaluate()

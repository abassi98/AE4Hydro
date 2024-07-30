import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    encoded_features_vec = [0,3,5,27]
    experiment = cfg["experiment"]
    initseed = 300
    nseeds = 4
    fig, ax = plt.subplots(2,2, sharey=True)
    # plot nse AE vs STAT colore with all the attributes
    for ind, encoded_features in enumerate(encoded_features_vec):
        j = ind % 2
        i = ind // 2
        df = pd.DataFrame()
        for seed in range(initseed, initseed+nseeds):
            # plot nse on the CONUS
            if encoded_features == 0:
                stats = pd.read_csv(f"analysis/stats/caam_{encoded_features}_{seed}.csv", sep=",")
            else:
                stats = pd.read_csv(f"analysis/stats/{experiment}_{encoded_features}_{seed}.csv", sep=",")
            stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
            nse = stats["nse"]
            df[seed] = nse

        df.index = np.arange(1,569)
        g = sns.heatmap(df.transpose().clip(-1,1), vmin=-1, vmax=1, cmap="viridis", ax=ax[i,j], annot=False)
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
    fig.savefig(f"analysis/figures/restarts.png", dpi=300)

    

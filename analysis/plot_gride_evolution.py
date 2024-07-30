
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__=="__main__":
    enc_vec = [2,3,5,27]
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    for ind, encoded_features in enumerate(enc_vec):
        i, j = ind//2, ind%2
        ax = axs[i,j]
        for j, seed in enumerate(range(300,304)):
            # read file
            df = pd.read_csv(f"analysis/encoded/id_enca_{encoded_features}_{seed}.txt")
            ax.scatter(df["d"], df["id"], label=f"Random Restart {j}", s=10)
            ax.grid(True, which='both', linestyle=':', linewidth=1, color='grey')
            ax.set_xscale('log')
            ax.tick_params(labelsize=20)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            
        ax.set_title(f"ENCA {encoded_features}", fontsize=30)
    handles, labels = ax.get_legend_handles_labels()
    #fig.legend(handles, labels,  bbox_to_anchor=(0.6, 0.55), fancybox=True, shadow=True)
    fig.supxlabel("Log10 of the average $n_2$ distance", fontsize=30)
    fig.supylabel("Intrinsic Dimension (ID)", fontsize=30)
    fig.tight_layout()
    fig.savefig("analysis/figures/gride_evolution.png", dpi=300)
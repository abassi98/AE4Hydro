import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from src.datautils import load_attributes
from pathlib import PosixPath
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 20 
mpl.rcParams['ytick.labelsize'] = 20 

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
    encoded_features_vec = [0,1,2,3,5,27]
    us_states = gpd.read_file('data/usa-states-census-2014.shp')
    experiment = cfg["experiment"]


    stat_caam = pd.read_csv(f"analysis/stats/{experiment}_stat_0.csv", sep=",")
    stat_caam.index = [str(s).rjust(8,"0") for s in stat_caam.loc[:,"basin"]]
    basins = stat_caam.index
    num_basins = stat_caam.shape[0]
    # load attributes
    camels_root = PosixPath("data/basin_dataset_public_v1p2/")
    #add_camels_attributes(camels_root)
    df_S = load_attributes("data/attributes.db", stat_caam.index, drop_lat_lon=False)
    # insert lat and lon in encoded attributes
    lon = df_S["gauge_lon"]
    lat = df_S["gauge_lat"]
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_S1 = load_attributes("data/attributes.db", stat_caam.index, drop_lat_lon=False, keep_features=True)
    df_S1= df_S1.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_S = pd.concat([df_S, df_S1], axis=1)

    

    markers = ["^", "o", "v", "<", ">", "s"]
    i=0
    line = np.linspace(-1,1,1000)
    # smooth nse copmarison
    sigma = 5

    
    
    # statistics distributions
    df_NSE = pd.DataFrame()
    df_BIAS = pd.DataFrame()
    df_STDEV = pd.DataFrame()
    df_FHV = pd.DataFrame()

    nse_caam= stat_caam["nse"]
    df_NSE["CAAM"] = nse_caam
    df_BIAS["CAAM"] = stat_caam["bias"]
    df_STDEV["CAAM"] = stat_caam["log_stdev"]
    
    

    # plot nse AE vs prev model
    nse_comp = nse_caam
    for i in range(len(encoded_features_vec)):
        # plot nse on the CONUS
        file_id = f"{experiment}_{encoded_features_vec[i]}"
        file_stat = f"analysis/stats/{file_id}.csv"
        stats = pd.read_csv(file_stat, sep=",")
        stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
        nse = stats["nse"]


        fig,ax = plt.subplots(1,2, figsize=(15,5),gridspec_kw={'width_ratios': [2, 1]})
        
        ax[0].set_xlim(-128, -65)
        ax[0].set_ylim(24, 50)
        ax[0].set_axis_off()
        us_states.boundary.plot(color="black", ax=ax[0], linewidth=0.5)
    
        color = np.clip(nse - nse_comp, -1,1)
        #mask = np.logical_or(np.logical_and(nse < 0.0, nse_comp < 0.0), np.logical_and(nse > 0.0,nse_comp > 0.0))
        #color[mask] = 0.0
        im = ax[0].scatter(x=lon, y=lat,c=color, s=100, cmap="coolwarm", vmin=-1,vmax=1 )
        if i==0:
            ax[0].set_title(f"Delta NSE: ENCA-{encoded_features_vec[i]} and CAAM", fontsize=35, y=-0.2)
            ax[1].set_ylabel("NSE CAAM", fontsize=35)
        else:
            ax[0].set_title(f"Delta NSE: ENCA-{encoded_features_vec[i]} and ENCA-{encoded_features_vec[i-1]}", fontsize=35, y=-0.2)
            ax[1].set_ylabel(f"NSE ENCA-{encoded_features_vec[i-1]}", fontsize=35)

            
        ax[1].set_xlabel(f"NSE ENCA-{encoded_features_vec[i]}", fontsize=35)
        ax[1].plot(line, line, c="black",ls="--")
        sc = ax[1].scatter(np.clip(nse, -1,1), np.clip(nse_comp,-1,1), c=color, s=100, cmap="coolwarm", vmin=-1,vmax=1 )
        cb = fig.colorbar(sc)
        cb.set_label("Delta NSE clipped in [-1,1]", fontsize=25) 
        ax[1].text(-0.8,0.8, f"r = {np.round(pearsonr(nse, nse_comp)[0], 2)}", fontsize=30)
        ax[1].grid()
        
        fig.legend()
        fig.tight_layout()
        save_file = f"analysis/figures/plot_delta_nse_{experiment}_{encoded_features_vec[i]}.png"
        fig.savefig(save_file)
        plt.close()
        


        ### retrieve NSE Distribution
        df_NSE[f"ENCA-{encoded_features_vec[i]}"] = stats["nse"]
        df_BIAS[f"ENCA-{encoded_features_vec[i]}"] = stats["bias"]
        df_STDEV[f"ENCA-{encoded_features_vec[i]}"] = stats["log_stdev"]
        

        # plot scatter delta nse vs attributes
        names_to_keep = ["baseflow_index", "aridity", "high_q_freq", "q95", "carbonate_rocks_frac", "elev_mean",
                         "frac_snow", "frac_forest", "soil_conductivity" ]
        
        fig_ar, ax_ar = plt.subplots(3,3, sharey=True)
        ind = 0
       
        # scatter plot of static features
        for key in df_S.keys():
            if key in names_to_keep:
                index = df_S[key]
                i1 = ind % 3
                i2  = ind // 3
                ax_ar[i1,i2].tick_params(axis='x', labelsize=5)
                ax_ar[i1,i2].tick_params(axis='y', labelsize=5)
                ax_ar[i1,0].set_ylabel("Delta NSE", fontsize=10,fontweight="bold")
                ax_ar[i1,i2].set_title(clean_and_capitalize(key), fontsize=10, fontweight="bold")
                ax_ar[i1,i2].grid()
                ax_ar[i1,i2].scatter(index,color, s=2, c="blue")
                ax_ar[i1,i2].axhline(0,c="orangered", ls="--", lw=2)
                ind += 1
            
        
        save_file = f"analysis/figures/attributes_nse/plot_DELTA_NSE_{experiment}_{encoded_features_vec[i]}.png"
        fig_ar.tight_layout()
        fig_ar.savefig(save_file)
        
        nse_comp = nse
    
    # plot NSE vs CAAM NSE
    for i in range(len(encoded_features_vec)):    
        # plot nse on the CONUS
        file_id = f"{experiment}_{encoded_features_vec[i]}"
        file_stat = f"analysis/stats/{file_id}.csv"
        stats = pd.read_csv(file_stat, sep=",")
        stats.index = [str(s).rjust(8,"0") for s in stats.loc[:,"basin"]]
        nse = stats["nse"]

        fig,ax = plt.subplots(1,2, figsize=(15,5),gridspec_kw={'width_ratios': [2, 1]})
        ax[0].set_xlim(-128, -65)
        ax[0].set_ylim(24, 50)
        ax[0].set_axis_off()
        us_states.boundary.plot(color="black", ax=ax[0], linewidth=0.5)
    
        color = np.clip(nse - nse_caam, -1,1)
        im = ax[0].scatter(x=lon, y=lat,c=color, s=100, cmap="coolwarm", vmin=-1,vmax=1 )
        ax[0].set_title(f"Delta NSE: ENCA-{encoded_features_vec[i]} and CAAM", fontsize=35, y=-0.2)
        ax[1].set_ylabel("NSE CAAM", fontsize=35)

            
        ax[1].set_xlabel(f"NSE ENCA-{encoded_features_vec[i]}", fontsize=35)
        ax[1].plot(line, line, c="black",ls="--")
        sc = ax[1].scatter(np.clip(nse, -1,1), np.clip(nse_caam,-1,1), c=color, s=100, cmap="coolwarm", vmin=-1,vmax=1 )
        cb = fig.colorbar(sc)
        cb.set_label("Delta NSE clipped in [-1,1]", fontsize=25) 
        ax[1].text(-0.8,0.8, f"r = {np.round(pearsonr(nse, nse_caam)[0], 2)}", fontsize=30)
        ax[1].grid()
        
        fig.legend()
        fig.tight_layout()
        save_file = f"analysis/figures/plot_delta_nse_{experiment}_{encoded_features_vec[i]}_vs_caam.png"
        fig.savefig(save_file)
        plt.close()
        
    mpl.rcParams['xtick.labelsize'] = 10 
    mpl.rcParams['ytick.labelsize'] = 10 

   

    ### plot metrics (NSE, BIAS,LOG-STDEV)
    fig, ax = plt.subplots(1,3, figsize=(25,8))
    colors = ["navy", "darkgreen",  "red", "chocolate", "darkorange", "orange", "goldenrod" ]
    ticks = np.arange(len(df_NSE.columns))
    labels = df_NSE.columns
    # NSE
    ax[0].grid()
    g = sns.boxplot(df_NSE,ax=ax[0], whis=(5,95))
    ax[0].set_xlabel(" ")
    ax[0].set_xticks(ticks, labels, fontsize=15)
    ax[0].set_ylabel("NSE", rotation=90, fontweight="bold", fontsize=20)
    ax[0].set_ylim([-1.0,1])
    # BIAS
    ax[1].grid()
    g = sns.boxplot(df_BIAS, ax=ax[1], whis=(5,95))
    ax[1].set_xlabel(" ")
    ax[1].set_xticks(ticks, labels, fontsize=15)
    ax[1].set_ylabel("BIAS", rotation=90, fontweight="bold", fontsize=20)
    ax[1].set_ylim([-1,2.0])
    # LOG-STDEV
    ax[2].grid()
    g = sns.boxplot(df_STDEV, ax=ax[2], whis=(5,95))
    ax[2].set_xlabel(" ")
    ax[2].set_xticks(ticks, labels, fontsize=15)
    ax[2].set_ylabel("LOG-STDEV", rotation=90, fontweight="bold", fontsize=20)
    ax[2].set_ylim([0,2])
    fig.tight_layout()
    fig.savefig("densities.png")

    stat_names = ["Median", "Mean", "Min", "Q5", "QS95", "Max"]
    stats = ["NSE", "BIAS", "STDEV"]
    # print latex table
    print("\\begin{table}[!t]")
    print("\\centering")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{l||ccccc|c}")
    print("& \\textbf{NO-STAT} & \\textbf{CAAM} & \\textbf{ENCA-2} &  \\textbf{ENCA-3} & \\textbf{ENCA-4} & \\textbf{ENCA-5} & \\textbf{ENCA-27}  \\\\")
    print("\\hline")

    df_NSE_des = df_NSE.describe(percentiles=[.05,.25, .5, .75, .95]).round(2).loc[["mean", "min", "5%", "25%", "50%", "75%","95%", "max"],:].rename({"mean":"  \\multirow{7}{c}{\\textbf{NSE}} & Mean", "min":"& Min","5%":"& Q5", "25%":"& Q25", "50%":"& Median", "75%":"& Q75", "95%":"& Q95", "max":"& Max"}, axis=0)
    df_BIAS_des = df_BIAS.describe(percentiles=[.05,.25, .5, .75, .95]).round(2).loc[["mean", "min", "5%", "25%", "50%", "75%","95%", "max"],:].rename( {"mean":"  \\multirow{7}{c}{\\textbf{BIAS}} & Mean", "min":"& Min","5%":"& Q5", "25%":"& Q25", "50%":"& Median", "75%":"& Q75", "95%":"& Q95", "max":"& Max"}, axis=0)
    df_STDEV_des = df_STDEV.describe(percentiles=[.05,.25, .5, .75, .95]).round(2).loc[["mean", "min", "5%", "25%", "50%", "75%","95%", "max"],:].rename({"mean":"  \\multirow{7}{c}{\\textbf{LOG-STDEV}} & Mean", "min":"& Min","5%":"& Q5", "25%":"& Q25", "50%":"& Median", "75%":"& Q75", "95%":"& Q95", "max":"& Max"}, axis=0)
    #df_FHV_des = df_FHV.describe(percentiles=[.05, .5, .95]).round(2).loc[["mean", "min", "5%", "50%", "95%", "max"],:].rename({"mean":" \\textbf{FHV} (Mean)", "min":"\\textbf{FHV} (Min)","5%":"\\textbf{FHV} (Q5)", "50%":"\\textbf{FHV} (Median)", "95%":"\\textbf{KGE} (Q95)", "max":"\\textbf{KGE} (Max)"}, axis=0)
   
    df = pd.concat((df_NSE_des, df_BIAS_des, df_STDEV_des), axis=0)
    latex_table = df.to_latex(escape=False, header=False)
    print(latex_table)

    print("\\end{tabular}}")
    print("\\end{table}")


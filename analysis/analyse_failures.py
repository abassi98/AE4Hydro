
import pickle
import matplotlib.pyplot as plt
from src.utils import get_basin_list
import pandas as pd
import numpy as np
from src.datautils import load_attributes

def evaluate():
    basins = get_basin_list()
    df_A = load_attributes("data/attributes.db", basins, drop_lat_lon=False, keep_features=None)
    df_S = load_attributes("data/attributes.db", basins, drop_lat_lon=False, keep_features=True)
    df_A = df_A.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_S = df_S.drop(['gauge_lat', 'gauge_lon'], axis=1)
    df_AS = pd.concat([df_A, df_S], axis=1)
    
    # retireve models
    encoded_features = [2,3,4,5,27]


    # load basin names
    info = pd.read_csv("data/basin_dataset_public_v1p2/basin_metadata/gauge_information.txt",engine="python",sep="\t")
    names = info.iloc[:,2]
    index = info.iloc[:,1]
    index = [str(b).rjust(8,"0") for b in index]
    
    names_dict = {}
    for i in range(len(names)):
        if index[i] in basins:
            names_dict[index[i]] = names[i]
    
   
    # dict to load selected cacthments info for each models
    improved_basins_models = {}
    worsened_basins_models = {}

    # access caam statistics
    df_caam = pd.read_csv("analysis/stats/global_lstm_ae_stat_0.csv", sep=",", index_col="basin")
    df_caam.index = [str(b).rjust(8,"0") for b in df_caam.index]
    df_enca3 = pd.read_csv(f"analysis/stats/global_lstm_ae_3.csv", sep=",", index_col="basin")
    df_enca3.index = [str(b).rjust(8,"0") for b in df_enca3.index]
    df_enca2 = pd.read_csv(f"analysis/stats/global_lstm_ae_2.csv", sep=",", index_col="basin")
    df_enca2.index = [str(b).rjust(8,"0") for b in df_enca2.index]

    # plot failure
    for i in range(len(encoded_features)):
        # load previous model
        if i==0:
            df_prev = df_caam
        else:
            df_prev = pd.read_csv(f"analysis/stats/global_lstm_ae_{encoded_features[i-1]}.csv", sep=",", index_col="basin")
            df_prev.index = [str(b).rjust(8,"0") for b in df_prev.index]

        # load model
        df_model = pd.read_csv(f"analysis/stats/global_lstm_ae_{encoded_features[i]}.csv", sep=",", index_col="basin")
        df_model.index = [str(b).rjust(8,"0") for b in df_model.index]

        # save basins
        improved_basins = []
        worsened_basins = []

        for basin in basins:
            if df_prev.loc[basin, "nse"] < 0.0 and  df_model.loc[basin, "nse"] > 0.0:
                # save basin
                improved_basins.append(basin)

            if df_prev.loc[basin, "nse"] > 0.0 and  df_model.loc[basin, "nse"] < 0.0:
                # save basin
                worsened_basins.append(basin)

            improved_basins_models[f"ENCA-{encoded_features[i]}"] = improved_basins
            worsened_basins_models[f"ENCA-{encoded_features[i]}"] = worsened_basins

        print(f"Number of improved basins in ENCA-{encoded_features[i]}: {len(improved_basins)}")
        print(f"Number of worsened basins in ENCA-{encoded_features[i]}: {len(worsened_basins)}")
        print("--------------------------------------------------------")

    # merge improved basins for different models without repetition
    improved_basins = list(set(improved_basins_models["ENCA-3"] + worsened_basins_models["ENCA-3"]))
    improved_basins = improved_basins_models["ENCA-3"]
    improved_basins.sort()
    print(improved_basins)
    # create data frame specific for the model
    df = pd.DataFrame(index=improved_basins)
    # add info
    df["Baseflow Index"] = df_AS.loc[improved_basins, "baseflow_index"].round(2)
    df["Aridity Index"] = df_AS.loc[improved_basins, "aridity"].round(2)
    df["Q95"] =  df_AS.loc[improved_basins, "q95"].round(2)
    df["High Flow Frequency"] = df_AS.loc[improved_basins, "high_q_freq"].round(2)
    df["BIAS (CAAM)"] = df_caam.loc[improved_basins,"bias"].round(2)
    df["BIAS (ENCA-2)"] = df_enca2.loc[improved_basins, "bias"].round(2)
    df["BIAS (ENCA-3)"] = df_enca3.loc[improved_basins, "bias"].round(2)
    df["LOG-STDEV (CAAM)"]  = df_caam.loc[improved_basins,"log_stdev"].round(2)
    df["LOG-STDEV (ENCA-2)"] = df_enca2.loc[improved_basins, "log_stdev"].round(2)
    df["LOG-STDEV (ENCA-3)"] = df_enca3.loc[improved_basins, "log_stdev"].round(2)
    df["NSE (CAAM)"]  = df_caam.loc[improved_basins,"nse"].round(2)
    df["NSE (ENCA-2)"] = df_enca2.loc[improved_basins, "nse"].round(2)
    df["NSE (ENCA-3)"] = df_enca3.loc[improved_basins, "nse"].round(2)
    latex_table = df.to_latex(escape=False, header=True)
    print(latex_table)
    

if __name__ == "__main__":
    evaluate()

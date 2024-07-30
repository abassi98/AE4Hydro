import pandas as pd
from ..src.utils import get_basin_list
from ..src.datautils import load_attributes, CLIM_NAMES, HYDRO_NAMES, LANDSCAPE_NAMES 

experiment = "enca"
threshold = 0.0
def evaluate():
    basins = get_basin_list()
    # retireve models
    encoded_features = [2,3,5,27]
    
    # load basin names
    info = pd.read_csv("../data/basin_dataset_public_v1p2/basin_metadata/gauge_information.txt",engine="python",sep="\t")
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
    df_caam = pd.read_csv(f"stats/caam_0.csv", sep=",", index_col="basin")
    df_caam.index = [str(b).rjust(8,"0") for b in df_caam.index]
    df_enca2 = pd.read_csv(f"stats/{experiment}_2.csv", sep=",", index_col="basin")
    df_enca2.index = [str(b).rjust(8,"0") for b in df_enca2.index]
    df_enca3 = pd.read_csv(f"stats/{experiment}_3.csv", sep=",", index_col="basin")
    df_enca3.index = [str(b).rjust(8,"0") for b in df_enca3.index]
    

    # plot failure
    for i in range(len(encoded_features)):
        # load previous model
        if i==0:
            df_prev = df_caam
        else:
            df_prev = pd.read_csv(f"stats/{experiment}_{encoded_features[i-1]}.csv", sep=",", index_col="basin")
            df_prev.index = [str(b).rjust(8,"0") for b in df_prev.index]

        # load model
        df_model = pd.read_csv(f"stats/{experiment}_{encoded_features[i]}.csv", sep=",", index_col="basin")
        df_model.index = [str(b).rjust(8,"0") for b in df_model.index]

        # save basins
        improved_basins = []
        worsened_basins = []

        for basin in basins:
            if df_prev.loc[basin, "nse"] < threshold and  df_model.loc[basin, "nse"] > threshold:
                # save basin
                improved_basins.append(basin)

            if df_prev.loc[basin, "nse"] > threshold and  df_model.loc[basin, "nse"] < threshold:
                # save basin
                worsened_basins.append(basin)

            improved_basins_models[f"ENCA-{encoded_features[i]}"] = improved_basins
            worsened_basins_models[f"ENCA-{encoded_features[i]}"] = worsened_basins

        print(f"Number of improved basins in ENCA-{encoded_features[i]}: {len(improved_basins)}")
        print(f"Number of worsened basins in ENCA-{encoded_features[i]}: {len(worsened_basins)}")
        print("--------------------------------------------------------")


if __name__ == "__main__":
    evaluate()

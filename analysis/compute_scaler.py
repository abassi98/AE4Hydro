
from tqdm import tqdm

import pandas as pd
import numpy as np

from pathlib import Path

import pandas as pd
from tqdm import tqdm
from typing import List

from src.datasets import CamelDataset
from torch.utils.data import DataLoader


#from LSTM_AE_main import GLOBAL_SETTINGS

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 2000,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 30,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'log_interval': 50,
    'learning_rate': 1e-3,
    'seq_length': 400,
    "linear" : 512,
    'train_start': pd.to_datetime('01101980', format='%d%m%Y'),
    'train_end': pd.to_datetime('30091995', format='%d%m%Y'),
    'val_start': pd.to_datetime('01101995', format='%d%m%Y'),
    'val_end': pd.to_datetime('29092010', format='%d%m%Y')
}
def get_basin_list() -> List:
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = Path(__file__).absolute().parent.parent / "MultiBasinHydro2/data/basin_list2.txt"
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins




def extract_statistics():
    basins = get_basin_list()
    max_x = np.zeros((568, 5))
    min_x = np.zeros((568, 5))
    max_y = np.zeros((568,))
    min_y = np.zeros((568,))
    i = 0

    total_x = np.zeros((0,5))
    total_y = np.zeros((0,1))

    ds = CamelDataset(
            camels_root=Path('data/basin_dataset_public_v1p2'),
            basins=basins,
            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
            is_train=False,
            total_length = 5478)
    
    loader =  DataLoader(ds, batch_size=1, shuffle=False)
    for data in tqdm(loader):
        # extarct data
        x, y = data
        x, y = x.squeeze().detach().numpy(), y.squeeze().unsqueeze(-1).detach().numpy()
       
        max_x[i] = np.amax(x, axis=0) 
        min_x[i] = np.amin(x, axis=0) 
        max_y[i] = np.amax(y, axis=0) 
        min_y[i] = np.amin(y, axis=0) 


        # assert that all runoff is greater than 0 and there are no nans
        assert(np.all(np.array(y.flatten()>=0)))
        assert(np.all(np.logical_not(np.isnan(y))))
        
        assert(x.shape[0]==y.shape[0])

        total_x = np.concatenate((total_x, x), axis=0)
        total_y = np.concatenate((total_y, y), axis=0)

        i += 1


    print("Min/(max-min) x (forcing): ", np.amin(total_x, axis=0), np.amax(total_x, axis=0)-np.amin(total_x, axis=0))
    print("Min/(max-min) y (streamflow): ", np.amin(total_y, axis=0), np.amax(total_y, axis=0)-np.amin(total_y, axis=0))

    print("Mean/std x (forcing): ", np.mean(total_x, axis=0), np.std(total_x, axis=0))
    print("Meam/std y (streamflow): ", np.mean(total_y, axis=0), np.std(total_y, axis=0))
    
if __name__ == "__main__":
    extract_statistics()


from tqdm import tqdm

import pandas as pd
import numpy as np

from pathlib import Path

import pandas as pd
from tqdm import tqdm
from typing import List

from src.datasets import CamelsTXT, CamelDataset
from torch.utils.data import DataLoader
from ..main.py import GLOBAL_SETTINGS
from ..src.utils import get_basin_list



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

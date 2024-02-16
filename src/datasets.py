
from pathlib import PosixPath
from typing import List, Tuple
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .datautils import (load_attributes, load_discharge, load_forcing, normalize_features)

class CamelDataset(Dataset):
    def __init__(self,
                 camels_root: PosixPath,
                 dates: List,
                 basins: List,
                 is_train: bool,
                 total_length: int = 5478,
                 db_path: str = None,
                 no_static: bool = True,
                ):
        super().__init__()
     
        self.camels_root = camels_root
        self.dates = dates
        self.basins = basins
        self.is_train = is_train
        self.total_length = total_length
        self.db_path = db_path
        self.no_static = no_static
        
        # load data
        self.attributes = self._load_attributes()

        self.x, self.y = self.load_data()

    def load_data(self, ):
        # run over basins
        x_data = torch.zeros((0, self.total_length,5))
        y_data = torch.zeros((0, self.total_length,1))

        for basin in tqdm(self.basins):
            # retrieve data
            df, area = load_forcing(self.camels_root, basin)
            df['QObs(mm/d)'] = load_discharge(self.camels_root, basin, area)

            start_date = self.dates[0] #- pd.DateOffset(days=self.warmup - 1)
            end_date = self.dates[1]
            df = df[start_date:end_date]

            # store first and last date of the selected period (including warm_start)
            self.period_start = df.index[0]
            self.period_end = df.index[-1]

            # use all meteorological variables as inputs
            x = np.array([
                df['PRCP(mm/day)'].values, df['SRAD(W/m2)'].values, df['Tmax(C)'].values,
                df['Tmin(C)'].values, df['Vp(Pa)'].values
                #            df['prcp(mm/day)'].values, df['srad(W/m2)'].values, df['tmax(C)'].values,
                #            df['tmin(C)'].values, df['vp(Pa)'].values
            ]).T

            y = np.array([df['QObs(mm/d)'].values]).T

            # normalize data, reshape for LSTM training and remove invalid samples
            x = normalize_features(x, variable='inputs',is_train=self.is_train)

            if self.is_train:
                # Deletes all records where no discharge was measured (-999)
                x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
                y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

                # Delete all samples, where discharge is NaN
                
                if np.sum(np.isnan(y)) > 0:
                    print(
                        f"Deleted {np.sum(np.isnan(y))} of {len(y)} records because of NaNs in basin {self.basin}"
                    )
                    x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                    y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

                # store std of discharge before normalization
                y = normalize_features(y, variable='output', is_train=self.is_train)


            # convert arrays to torch tensors
            x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
            y = torch.from_numpy(y.astype(np.float32)).unsqueeze(0)

            # concatente
            x_data = torch.cat((x_data, x), dim=0)
            y_data = torch.cat((y_data, y), dim=0)
            
        
        return x_data, y_data
            
       
    def _load_attributes(self) -> torch.Tensor:
        df = load_attributes(self.db_path, self.basins, drop_lat_lon=True)
        
        # store means and stds
        self.attribute_means = df.mean()
        self.attribute_stds = df.std()


        # normalize data    
        df = (df - self.attribute_means) / self.attribute_stds
    

        # store attribute names
        self.attribute_names = df.columns

        # store feature as PyTorch Tensor
        attributes = df.values
        return torch.from_numpy(attributes.astype(np.float32))
    
    
        
    def __len__(self):
        return len(self.basins)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.attributes[idx]




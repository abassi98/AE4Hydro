"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019). 
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065 

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""
import sys
from pathlib import Path, PosixPath
from typing import List
import warnings 
import h5py
import numpy as np
import argparse
from tqdm import tqdm
from pytorch_lightning import Callback
import os 
import torch
import copy

from .datasets import CamelsTXT
from torch.optim.lr_scheduler import _LRScheduler



def create_h5_files(camels_root: PosixPath,
                    out_file: PosixPath,
                    basins: List,
                    dates: List,
                    with_basin_str: bool = True,
                    seq_length: int = 270,
                    to_encode_length: int = 1000):
    """[summary]
    
    Parameters
    ----------
    camels_root : PosixPath
        Path to the main directory of the CAMELS data set
    out_file : PosixPath
        Path of the location, where the hdf5 file should be stored
    basins : List
        List containing the 8-digit USGS gauge id
    dates : List
        List of start and end date of the discharge period to use, when combining the data.
    with_basin_str : bool, optional
        If True, stores for each sample the corresponding USGS gauged id, by default True
    seq_length : int, optional
        Length of the requested input sequences., by default 270
    to_encode_length : int, optional
        Length of the output sequence to pass to the decoder, by default two years
    
    Raises
    ------
    FileExistsError
        If file at this location already exists.
    """
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")

    with h5py.File(out_file, 'w') as out_f:
        input_data = out_f.create_dataset(
            'input_data',
            shape=(0, seq_length, 5),
            maxshape=(None, seq_length, 5),
            chunks=True,
            dtype=np.float32,
            compression='gzip')
        target_data = out_f.create_dataset(
            'target_data',
            shape=(0, seq_length, 1),
            maxshape=(None, seq_length, 1),
            chunks=True,
            dtype=np.float32,
            compression='gzip')
        

        q_stds = out_f.create_dataset(
            'q_stds',
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.float32,
            compression='gzip',
            chunks=True)

        if with_basin_str:
            sample_2_basin = out_f.create_dataset(
                'sample_2_basin',
                shape=(0,),
                maxshape=(None,),
                dtype="S10",
                compression='gzip',
                chunks=True)
        num_basins = 0
        for basin in tqdm(basins, file=sys.stdout):

            dataset = CamelsTXT(
                camels_root=camels_root,
                basin=basin,
                is_train=True,
                seq_length=seq_length,
                to_encode_length=to_encode_length,
                dates=dates)
            
            num_basins += 1
            num_samples = len(dataset)
            total_samples = input_data.shape[0] + num_samples

            # store input and output samples
            input_data.resize((total_samples, seq_length, 5))
            target_data.resize((total_samples, seq_length, 1))
            input_data[-num_samples:, :, :] = dataset.x
            target_data[-num_samples:, :, :] = dataset.y

            # # store to_encode_length for each basin
           

            # additionally store std of discharge of this basin for each sample
            q_stds.resize((total_samples, 1))
            q_std_array = np.array([dataset.q_std] * num_samples, dtype=np.float32).reshape(-1, 1)
            q_stds[-num_samples:, :] = q_std_array

            if with_basin_str:
                sample_2_basin.resize((total_samples,))
                str_arr = np.array([basin.encode("ascii", "ignore")] * num_samples)
                sample_2_basin[-num_samples:] = str_arr

            out_f.flush()


def get_basin_list() -> List:
    """Read list of basins from text file.
    
    Returns
    -------
    List
        List containing the 8-digit basin code of all basins
    """
    basin_file = Path(__file__).absolute().parent.parent / "data/basin_list.txt" 
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    basins = [basin.strip() for basin in basins]
    return basins



class MetricsCallback(Callback):
    """
    PyTorch Lightning metric callback.
    Save logged metrics
    """

    def __init__(self, dirpath, filename):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.path = os.path.join(dirpath, filename)
        exists = os.path.exists(self.path)
        # if already exists a saving, load it and update
        if exists:
            self.dict_metrics = torch.load(self.path, map_location=torch.device('cpu'))
        else:
            os.makedirs(self.dirpath, exist_ok = True) 
            self.dict_metrics = {}
            
        
    def on_validation_epoch_end(self,trainer, pl_module):
        epoch_num = int(trainer.logged_metrics["epoch_num"].cpu().item())
        self.dict_metrics["Epoch: "+str(epoch_num)] = copy.deepcopy(trainer.logged_metrics)
        torch.save(self.dict_metrics, self.path)



class AdaptiveScheduler(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the
    number of epoch reaches one of the milestones. Notice that such decay can
    happen simultaneously with other changes to the learning rate from outside
    this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (dict): Dictionary of epoch indices/learning rates. Keys must be increasing.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.002    if 30 <= epoch < 80
        >>> # lr = 0.0004   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones={30 : 0.002 , 80 :0.0004})
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, milestones, last_epoch=-1, verbose=False):
        self.optimizer = optimizer
        self.milestones = milestones
        super().__init__(optimizer, last_epoch, verbose)

 
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch not in self.milestones.keys():
            return [group['lr'] for group in self.optimizer.param_groups]
        return [self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]


def find_best_epoch(dirpath):
    """
    Find the epoch at which the validation error is minimized, or quivalently
    when thevalidation NSE is maximized
    Returns
    -------
        best_epoch : (int)
    """
    path_metrics = os.path.join(dirpath, "metrics.pt")
    data = torch.load(path_metrics, map_location=torch.device('cpu'))
    epochs_mod = []
    nse_mod = []
    for key in data:
        epoch_num = data[key]["epoch_num"]
        nse = -data[key]["val_loss"]
        if isinstance(epoch_num, int):
            epochs_mod.append(epoch_num)
            nse_mod.append(nse)
        else:
            epochs_mod.append(int(epoch_num.item()))
            nse_mod.append(nse.item())
    idx_ae = np.argmax(nse_mod)
    return int(epochs_mod[idx_ae])


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
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
        if w=="carbonate":
            w="Carb."
        if w=="permeability":
            w="Perm."
        if w=="porostiy":
            w="Porosity"
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

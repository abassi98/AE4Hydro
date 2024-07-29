

import argparse
import json
import pickle
import random
import glob
from collections import defaultdict
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger


from src.datasets import CamelDataset
from src.datautils import add_camels_attributes, rescale_features, normalize_features
#from papercode.models import Hydro_LSTM
from src.nseloss import NSELoss
from src.utils import get_basin_list, str2bool

from src.models import Hydro_LSTM_AE



###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 64,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.4,
    'epochs': 10000,
    'hidden_size': 256,
    'initial_forget_gate_bias': 5,
    'num_lstm_layers' : 1,
    'bidirectional' : False,
    'log_interval': 50,
    'learning_rate': 1e-5,
    'total_length': 5478,
    'warmup' : 270,
    "linear" : 512,
    'train_start': pd.to_datetime('01101980', format='%d%m%Y'),
    'train_end': pd.to_datetime('30091995', format='%d%m%Y'),
    'test_start': pd.to_datetime('01101995', format='%d%m%Y'),
    'test_end': pd.to_datetime('29092010', format='%d%m%Y')
}

def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate"])
    parser.add_argument('--name', type=str, default="benchmark", help="Name of the experiment.")

    parser.add_argument(
        '--camels_root',
        type=str,
        default='data/basin_dataset_public_v1p2/',
        help="Root directory of CAMELS data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help="User-selected GPU ID - if none chosen, will default to cpu")
    
    parser.add_argument(
        '--num_workers', type=int, default=3, help="Number of parallel threads for data loading")
    
    parser.add_argument(
        '--encoded_features',
        type=int,
        default=27,
        help="Dimension of the Encoder latent space.")
  
   
    parser.add_argument(
        '--basin_file',
        type=str,
        default=None,
        help="Path to file containing usgs basin ids. Default is data/basin_list.txt")
    
    parser.add_argument(
        '--no_static',
        type=str2bool,
        default=True,
        help="If True, trains LSTM without static features")
   

    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] in ["train"]) and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

   
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    if cfg["camels_root"] is not None:
        cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
  

    # initialise wandb runs
    if cfg["no_static"]:
        wandb.init(name=cfg["name"], project="AE4Hydro", dir=f"runs/ENCA_{cfg['encoded_features']}")
    else:
        wandb.init(name=cfg["name"], project="AE4Hydro", dir=f"runs/CAAM_{cfg['encoded_features']}")

    cfg['run_dir'] = Path(__file__).absolute().parent / wandb.run.dir
    cfg["train_dir"] = cfg["run_dir"] / 'data' 
    cfg["train_dir"].mkdir(parents=True)
    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, train_basins: List, val_basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    train_basins : List
        List containing the 8-digit USGS gauge id used for training
    val_basins : List
        List containing the 8-digit USGS gauge id used for validation 
    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    return cfg



###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # get basins
    train_basins = get_basin_list()
    test_basins = get_basin_list()

    # create folder structure for this run
    cfg = _setup_run(cfg)
    
    # prepare static atributes
    cfg = _prepare_data(cfg=cfg, train_basins=train_basins, val_basins=val_basins)

    # prepare PyTorch DataLoader for training
    ds_train = CamelDataset(
        camels_root=cfg["camels_root"],
        dates = [cfg["train_start"], cfg["train_end"]],
        basins=train_basins,
        is_train=True,
        total_length=cfg["total_length"],
        db_path=cfg["db_path"],
        no_static=cfg["no_static"],
        )
    
    train_loader = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    # prepare PyTorch DataLoader for validation
    ds_test = CamelDataset(
        camels_root=cfg["camels_root"],
        dates = [cfg["test_start"], cfg["test_end"]],
        basins=test_basins,
        is_train=True,
        total_length=cfg["total_length"],
        db_path=cfg["db_path"],
        no_static=cfg["no_static"],
        )
    
    test_loader = DataLoader(ds_test, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

  
    ### Pytorch Lightning model
    milestones = {0: 1e-5}
    model = Hydro_LSTM_AE(
                 in_channels = (1, 8, 16),
                 out_channels = (8, 16, 32),
                 kernel_sizes =  (7, 5, 2), 
                 padding = (0,0,0),
                 encoded_space_dim = cfg["encoded_features"],
                 lstm_hidden_units =  cfg["hidden_size"], 
                 bidirectional = cfg["bidirectional"],
                 initial_forget_bias=cfg["initial_forget_gate_bias"],
                 layers_num = cfg["num_lstm_layers"],
                 act = nn.LeakyReLU, 
                 loss_fn = nn.MSELoss(),
                 drop_p = cfg["dropout"], 
                 warmup = cfg["warmup"],
                 seq_length = cfg["total_length"],
                 lr = cfg["learning_rate"],
                 linear = cfg["linear"],
                 input_size_dyn = 5,
                 milestones = milestones,
                 no_static=cfg["no_static"],
                 )
    
    # override and load from checkpoint to continue training
    #model = Hydro_LSTM_AE.load_from_checkpoint(f"{cfg['run_dir']}/last.ckpt")
    print(model)
    logger = WandbLogger()
   

    checkpoint_model = ModelCheckpoint(
            save_top_k=1,
            save_last=True,
            monitor="val_loss",
            mode="min",
            dirpath= cfg["run_dir"],
            filename= "model-{epoch:02d}",
        )

    # define trainer 
    torch.set_float32_matmul_precision('high')
    
    if cfg["clip_norm"]:
        trainer = pl.Trainer(max_epochs=cfg["epochs"], callbacks=[checkpoint_model], accelerator=str(DEVICE), devices=1, logger=logger, gradient_clip_val=cfg["clip_value"])
    else:
        trainer = pl.Trainer(max_epochs=cfg["epochs"], callbacks=[checkpoint_model], accelerator=str(DEVICE), devices=1, logger=logger)

    torch.cuda.empty_cache()
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
   

def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    basins = get_basin_list()

    # get attribute means/stds from trainings dataset
    #train_file = user_cfg["run_dir"] / "data/train_data.h5"
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    df_failures = pd.read_csv("failures_nse.csv", sep=" ").iloc[:,1:]
    #ds_train = CamelsH5(
        #h5_file=train_file, db_path=db_path, basins=basins, concat_static=run_cfg["concat_static"])
    #means = ds_train.get_attribute_means()
    #stds = ds_train.get_attribute_stds()


    # model name
    if run_cfg["no_static"] == False:
        model_name = "CAAM"
    else:
        model_name = f"ENCA-{run_cfg['encoded_features']}"
  
  
    ### Pytorch Lightning model
    model = Hydro_LSTM_AE(
                 in_channels = (1, 8, 16),
                 out_channels = (8, 16, 32),
                 kernel_sizes =  (7, 5, 2), 
                 padding = (0,0,0),
                 encoded_space_dim = run_cfg["encoded_features"],
                 lstm_hidden_units =  run_cfg["hidden_size"], 
                 bidirectional = run_cfg["bidirectional"],
                 initial_forget_bias=run_cfg["initial_forget_gate_bias"],
                 layers_num = run_cfg["num_lstm_layers"],
                 act = nn.LeakyReLU, 
                 loss_fn = nn.MSELoss(),
                 drop_p = run_cfg["dropout"], 
                 warmup = run_cfg["warmup"],
                 seq_length = run_cfg["total_length"],
                 lr = run_cfg["learning_rate"],
                 linear = run_cfg["linear"],
                 input_size_dyn = 5,
                 no_static=run_cfg["no_static"],
                )
    
    # load model
    weight_file =  glob.glob(f"{user_cfg['run_dir']}/last.ckpt")[0]
    print("Analaysing model at ", weight_file)
    p = torch.load(weight_file, map_location=torch.device("cpu"))
    model.load_state_dict(p["state_dict"])
    
    # set data range
    date_range = pd.date_range(start=GLOBAL_SETTINGS["test_start"] + pd.offsets.Day(run_cfg["warmup"]), end=GLOBAL_SETTINGS["test_end"])
    results = {}
    results_enc = {}


    # initialize test dataset
    ds_test = CamelDataset(
            camels_root=Path(run_cfg["camels_root"]),
            dates = [pd.to_datetime(run_cfg["test_start"], format='%d%m%Y'), pd.to_datetime(run_cfg["test_end"], format='%d%m%Y')],
            basins=basins,
            is_train=False,
            db_path = db_path,
            no_static=run_cfg["no_static"])
    
    loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=0)
    for basin, batch in tqdm(zip(basins, loader)):
        if model.encoded_space_dim > 0:
            preds, obs, enc = evaluate_basin(model, batch)
            columns = [f"enc_{i+1}" for i in range(run_cfg["encoded_features"])] 
            df_enc = pd.DataFrame(data=enc, index=columns)
            results_enc[basin] = df_enc
        else:
            preds, obs  = evaluate_basin(model, batch)

        preds = preds[:,run_cfg["warmup"]:,:]
        obs = obs[:,run_cfg["warmup"]:,:]
        
        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)
        results[basin] = df
        
    if model.encoded_space_dim > 0:
        _store_results(user_cfg, run_cfg, results_enc, enc=True)
        
    _store_results(user_cfg, run_cfg, results)
    


def evaluate_basin(model: nn.Module, batch: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to train
    loader : DataLoader
        PyTorch DataLoader containing the basin data in batches.

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) network prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    model.eval()
    preds, obs, enc = None, None, None

    with torch.no_grad():
        x, y, attr = batch            
        #print(y_to_encode)
        x, y, attr = x.to(DEVICE), y.to(DEVICE), attr.to(DEVICE)
        # y = y.unsqueeze(0)
        #y_norm = normalize_features(y, variable='output').float()
        y = normalize_features(y, variable="output").float()
        e, p = model(x, y, attr)
        
        if preds is None:
            preds = p.detach().cpu()
            obs = y.detach().cpu()
            if model.encoded_space_dim > 0:
                enc = e.detach().cpu()
               
        else:
            preds = torch.cat((preds, p.detach().cpu()), 0)
            obs = torch.cat((obs, y.detach().cpu()), 0)
            if model.encoded_space_dim > 0:
                enc = torch.cat((enc, e.detach().cpu()), 0)
               
        preds = rescale_features(preds.numpy(), variable='output')
        obs = rescale_features(obs.numpy(), variable="output")
        
        # set discharges < 0 to zero
        preds[preds < 0] = 0

    if model.encoded_space_dim > 0:
        enc = enc.flatten().numpy()
        return preds, obs, enc
    else:
        return preds, obs




def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame, enc =False):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    if enc:
        file_name = user_cfg["run_dir"] / f"encoded_{run_cfg['encoded_features']}_seed{run_cfg['seed']}.p"
        with (file_name).open('wb') as fp:
            pickle.dump(results, fp)
    else:
        file_name = user_cfg["run_dir"] / f"lstm_ae_static_seed{run_cfg['seed']}.p"
        with (file_name).open('wb') as fp:
            pickle.dump(results, fp)

    print(f"Sucessfully store results at {file_name}")



if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)

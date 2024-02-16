#!/bin/bash

#SBATCH --job-name=gpu_LSTM_global
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=30:00:00
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM_global.out
#SBATCH --error=gpu_LSTM_global.err
#SBATCH --account=em09


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load daint-gpu PyTorch


# the above code is meant to run on ZAHW cluster in Zurich. Modify it to use in your machine/cluster. 
# from here onwards you can run everywhere

source ./hydro/bin/activate
wandb offline

nseeds=4
firstseed=300
gpu=0

for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
    
    if [ "$1" = "lstm_ae" ]
    then
      if [ ! -d "reports/LSTM_AE_$2/" ] 
      then
        mkdir reports/LSTM_AE_$2/
        mkdir runs/LSTM_AE_$2/
      fi
      outfile="reports/LSTM_AE_$2/global_lstm_ae_nldas.$seed.out"
      python LSTM_AE_main.py --name="LSTM/AE-$2/global" --gpu=$gpu --use_mse=True --encoded_features=$2 train > $outfile
    elif [ "$1" = "lstm_ae_stat" ]
    then
      if [ ! -d "reports/LSTM_AE_$2/" ] 
      then
        mkdir reports/LSTM_AE_$2/
        mkdir runs/LSTM_AE_$2/
      fi
      outfile="reports/LSTM_AE_$2/global_lstm_ae_stat_nldas.$seed.out"
      python LSTM_AE_main.py --no_static=False --name="LSTM/AE-STAT-$2-$seed/global" --gpu=$gpu --use_mse=True --encoded_features=$2 train > $outfile  
    else
      echo bad model choice
      exit
    fi
  
done
    
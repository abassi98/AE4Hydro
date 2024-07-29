#!/bin/bash

#SBATCH --job-name=gpu_LSTM_global
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --time=15:00:00
#SBATCH --partition=earth-4
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=gpu_LSTM_global.out
#SBATCH --error=gpu_LSTM_global.err

module load gcc/9.4.0-pe5.34 miniconda3/4.12.0 lsfm-init-miniconda/1.0.0	# comment to run on your machine

# the above code is meant to run on ZAHW cluster in Zurich. Modify it to use in your machine/cluster. 
# from here onwards you can run everywhere

conda activate hydro # comment to run on your machine
wandb offline

nseeds=4
firstseed=300
gpu=0

for (( seed = $firstseed ; seed < $((nseeds+$firstseed)) ; seed++ )); do
    if [ "$1" = "enca" ]
    then
      if [ ! -d "reports/ENCA_$2/" ] 
      then
        mkdir reports/ENCA_$2/
        mkdir runs/ENCA_$2/
      fi
      outfile="reports/ENCA_$2/enca_nldas_ext.$seed.out"
      python main.py --no_static=True --name="ENCA-$2-$seed" --gpu=$gpu --encoded_features=$2 train > $outfile
    elif [ "$1" = "caam" ]
    then
      if [ ! -d "reports/CAAM_$2/" ] 
      then
        mkdir reports/CAAM_$2/
        mkdir runs/CAAM_$2/
      fi
      outfile="reports/CAAM_$2/caam_nldas.$seed.out"
      python main.py --no_static=False --name="CAAM-$2-$seed" --gpu=$gpu --encoded_features=$2 train > $outfile  
    else
      echo bad model choice
      exit
    fi
done
    
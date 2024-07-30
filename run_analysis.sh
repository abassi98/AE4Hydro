#!/bin/bash

#SBATCH --job-name=run_stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --begin=now 
#SBATCH --partition=earth-3
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END,FAIL     
#SBATCH --output=run_stat.out
#SBATCH --error=run_stat.err

module load gcc/9.4.0-pe5.34 miniconda3/4.12.0 lsfm-init-miniconda/1.0.0	


# the above code is meant to run on ZAHW cluster in Zurich. Modify it to use in your machine/cluster. 
# from here onwards you can run everywhere

conda activate hydro

if [ "$1" = "enca" ]
then
    python run_analysis.py ENCA_$2 $1 $2
    python analysis/main_performance.py $1 $2
elif [ "$1" = "caam" ]
then
    python run_analysis.py CAAM_$2 $1 $2
    python analysis/main_performance.py $1 $2





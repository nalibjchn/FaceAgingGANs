#!/bin/bash -l
#SBATCH --job-name=age
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue for computer science

#SBATCH --partition=csgpu
# specify one GPU server
#SBATCH --gres=gpu:1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 10

# specify the walltime e.g 20 mins
#SBATCH -t 60:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxx@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR
cd Face_Age_Gender_Estimator
module load tensorflowgpu
python face_age_estimator_wiki_imdb.py
# command to use

#!/bin/bash -l
#SBATCH --job-name=CAAEfromscratch
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue for computer science

#SBATCH --partition=csgpu
# specify one GPU server
#SBATCH --gres=gpu:1
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 15

# specify the walltime e.g 20 mins
#SBATCH -t 96:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=xxxx@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR
cd Face_Aging_CAAE_10age
module load tensorflowgpu
python main.py --is_train True --dataset ../DATA/TrainingSet_CACD2000 --savedir save --use_trained_model false --use_init_model false
# command to use

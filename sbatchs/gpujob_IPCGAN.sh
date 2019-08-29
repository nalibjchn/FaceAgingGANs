#!/bin/bash -l
#SBATCH --job-name=IPCGAN
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
#SBATCH --mail-user=xxx@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR
cd IPCGAN_Face_Aging_5AgeGroups
module load tensorflowgpu
python python pre_trainedmodel_test.py --test_data_dir=../DATA/TestSet_FGNET --root_folder=../DATA/TrainingSet_CACD2000/

# command to use

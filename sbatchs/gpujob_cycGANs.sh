#!/bin/bash -l
#SBATCH --job-name=FaceVcycGAN
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
cd -P CycleGANs/CycleGANs_v1
module load tensorflowgpu
python faceaging_cyclegan.py --root_path ../../DATA/CycleGANs_Paired_TrainingSet
# command to use

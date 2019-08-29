#!/bin/bash -l
#SBATCH --job-name=ageestimatorcup
# speficity number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 35

# specify the walltime e.g 20 mins
#SBATCH -t 36:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

# run from current directory
cd $SLURM_SUBMIT_DIR
cd Face_Age_Gender_Estimator
python face_age_estimator_wiki_imdb.py
# command to use
hostname

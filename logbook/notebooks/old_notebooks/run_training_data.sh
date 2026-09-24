#!/bin/bash
#PBS -N training_data
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=12:00:00

module load Python/3.12.3-GCCcore-13.3.0
source ~/envs/wake_env/bin/activate

cd /rds/general/user/vk825/home/wake_effect/irp-vk825/notebooks

jupyter nbconvert \
    --to notebook \
    --execute training_data_generation.ipynb \
    --output training_data_generation_executed.ipynb


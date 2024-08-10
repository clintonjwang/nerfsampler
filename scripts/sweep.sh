#! /bin/bash
#SBATCH --partition=gpu
#SBATCH --time=0
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --exclude=zaatar,anise,mint,clove,sumac
#SBATCH -e /data/vision/polina/users/clintonw/code/inrnet/results/_/%A_%a.err
#SBATCH -o /data/vision/polina/users/clintonw/code/inrnet/results/_/%A_%a.out

cd /data/vision/polina/users/clintonw/code/inrnet/nerfsampler
python train.py -j=${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID} -c=$conf --sweep_id=$SLURM_JOB_NAME

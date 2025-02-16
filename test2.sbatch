#!/bin/bash

#SBATCH --account pr_100_tandon_advanced
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h100:2
#SBATCH --job-name=collate-and-evolve
#SBATCH --mail-user=axs10302@nyu.edu
#SBATCH --time=30:00:00
#SBATCH --output=stdout.out

RUNDIR=$SCRATCH/llama3/

module purge 

singularity exec --nv --overlay /scratch/axs10302/pytorch-example/my_pytorch.ext3:r /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c " source /ext3/env.sh;
torchrun main.py"

echo "Process $SLURM_PROCUD of Job $SLURM_JOBID with the local id
 $SLURM_LOCALID using gpu id $CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on 
$(hostname))" 
echo "computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_dEVICE | tail -n 1)" 
sleep 15 





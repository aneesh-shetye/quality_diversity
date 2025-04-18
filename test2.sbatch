#!/bin/bash

#SBATCH --account pr_317_tandon_priority
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=quality-diversity
#SBATCH --mail-user=axs10302@nyu.edu
#SBATCH --time=2:00:00
#SBATCH --output=stdout.out

RUNDIR=$SCRATCH/llama3/

module purge 

# Log host memory usage every 10 seconds
while true; do
    echo "=== $(date) ==="
    free -h
    echo ""
    sleep 10
done > logs/ram_usage.log &

# Log per-process memory usage every 10 seconds
while true; do
    echo "=== $(date) - Top Processes by Memory ==="
    ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%mem | head -n 20
    echo ""
    sleep 10
done > process_memory.log &


# Capture the background process PID to terminate it later if needed
RAM_MONITOR_PID=$!
PROC_MONITOR_PID=$!

singularity exec --nv --overlay /scratch/axs10302/pytorch-example2/my_pytorch.ext3:r /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c " source /ext3/env.sh;
torchrun main.py"

nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free \
         --format=csv -l 10 > logs/gpu_usage_${SLURM_JOB_ID}.log &

echo "Process $SLURM_PROCUD of Job $SLURM_JOBID with the local id
 $SLURM_LOCALID using gpu id $CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on 
$(hostname))" 
echo "computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_dEVICE | tail -n 1)" 
sleep 15 

kill $RAM_MONITOR_PID $PROC_MONITOR_PID


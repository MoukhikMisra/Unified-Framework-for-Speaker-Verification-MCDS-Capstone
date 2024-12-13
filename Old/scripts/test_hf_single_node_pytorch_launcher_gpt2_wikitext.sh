#!/bin/bash
#SBATCH --account=bcey-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --gpus-per-node=4
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --mem-per-gpu=60G
#SBATCH --output=/scratch/bcey/hchen10/haoc/sbatch_logs/%x-%j.out
#SBATCH --err=/scratch/bcey/hchen10/haoc/sbatch_logs/%x-%j.err
#SBATCH --job-name=test_hf_single_node_pytorch_launcher_gpt2_wikitext

set -x -e

nvidia-smi
pwd

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
source /u/hchen10/miniconda3/etc/profile.d/conda.sh
conda activate llm
cd /scratch/bcey/hchen10/haoc/llm/transformers/examples/pytorch/language-modeling

# have the below in case of debugging nccl issues such as nccl timeout.
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO
# hide duplicated errors using this hack - will be properly fixed in pt-1.12
# export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json

# force crashing on nccl issues like hanging broadcast
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

echo "START TIME: $(date)"

# CHANGE TO CUMMULATIVELY LOG OUTPUTS
LOG_PATH="main_log.txt"
GPUS_PER_NODE=4
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NUM_PROCESSES=$NUM_PROCESSES GPUS_PER_NODE=$GPUS_PER_NODE NNODES=$NNODES \$SLURM_PROCID"

export HF_HOME=/scratch/bcey/hchen10/haoc/huggingface_cache
export HF_DATASETS_CACHE=/scratch/bcey/hchen10/haoc/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/bcey/hchen10/haoc/huggingface_cache
# export WANDB_PROJECT=TSLM
# export WANDB_ENTITY=cmu-mlsp-emo

# OTHER LAUNCHERS CAN BE USED HERE
# export LAUNCHER="deepspeed \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --launcher slurm \
#     --num_gpus $GPUS_PER_NODE \
#     "
    # --ddp_timeout 2400 \
export LAUNCHER="python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank \$SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT "

export PROGRAM="\
    run_clm.py \
        --model_name_or_path gpt2 \
        --dataset_name wikitext \
        --dataset_config_name wikitext-103-raw-v1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --output_dir /scratch/bcey/hchen10/haoc/llm/experiments/test_hf_single_node_pytorch_launcher_gpt2_wikitext \
        --max_train_samples 1000 \
    "


export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"
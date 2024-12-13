#!/bin/bashd
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --job-name=mrpc-1gpu-bs1
#SBATCH --output=/ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling/scripts/logs/%x-%j.out
#SBATCH--err=/ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling/scripts/logs/%x-%j.err

set -x -e

nvidia-smi
pwd

# CHANGE HERE THE CONDA EVN AND ANY STARTUP SCRIPTS
source /jet/home/mmisra/miniconda3/etc/profile.d/conda.sh
conda activate benchmark
cd /ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling

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
LOG_PATH="/ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling/scripts/logs/log$SLURM_JOBID.txt"
GPUS_PER_NODE=1
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NUM_PROCESSES=$NUM_PROCESSES GPUS_PER_NODE=$GPUS_PER_NODE NNODES=$NNODES \$SLURM_PROCID"

export HF_HOME=/ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling/scripts/hg_cache
export HF_DATASETS_CACHE=/ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling/scripts/hg_cache
export TRANSFORMERS_CACHE=/ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling/scripts/hg_cache
#export WANDB_PROJECT=TSLM
#export WANDB_ENTITY=cmu-mlsp-emo

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
    run_glue_no_trainer.py \
        --model_name_or_path datajuicer/LLaMA-1B-dj-refine-100B \
        --task_name mrpc \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --output_dir /ocean/projects/cis220031p/mmisra/transformers/examples/pytorch/language-modeling/scripts/experiments/mrpcds8 \
    "


#Check the memory usage
#Try using Zero (Deep Speed)
export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"

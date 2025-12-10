#!/bin/bash

models=(
    huggyllama/llama-7b
    # Qwen/Qwen3-4B
)

datasets=(
    <YOUR_PATH>/train_data.json
)

project="LongLoRA"
image="pytorch_rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        model_name=$(basename "$model")
        dataset_name=$(basename "$dataset" .json)
        timestamp=$(date +%Y%m%d_%H%M%S)

        exp_name="${timestamp}_${model_name}_${dataset_name}"

        script="tmp_run_${exp_name}.sh"
        cat > "$script" <<EOF
#!/bin/bash -l
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --time=03:00:00
#SBATCH --job-name=${exp_name}
#SBATCH --output=<YOUR_PATH>/slurm-%x-%j.out

export OUTPUT_ROOT="\${MYSCRATCH}/LongLoRA/output"
export EXP_DIR="\${OUTPUT_ROOT}/${exp_name}"
export LOG_FILE="\${EXP_DIR}/log.txt"
export CONFIG_FILE="\${EXP_DIR}/config.yaml"

mkdir -p "\$EXP_DIR"
cp "\$0" "\${EXP_DIR}/run_script.sh"

cat > "\$CONFIG_FILE" <<YAML
model_name_or_path: ${model}
data_path: ${dataset}
output_dir: \${EXP_DIR}
bf16: true
model_max_length: 1024
use_flash_attn: true
low_rank_training: false
num_train_epochs: 10
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
gradient_accumulation_steps: 8
save_strategy: steps
save_steps: 10
learning_rate: 2.0e-05
weight_decay: 0.0
warmup_steps: 20
lr_scheduler_type: constant_with_warmup
logging_steps: 1
deepspeed: ds_configs/stage3.json
tf32: false
YAML

cd \$MYSCRATCH/LongLoRA

GIT_HASH=\$(git rev-parse --short HEAD)
HOST_NAME=\$(hostname)
echo "==================================================" > \$LOG_FILE
echo "Experiment: ${exp_name}" >> \$LOG_FILE
echo "Start Time: \$(date)" >> \$LOG_FILE
echo "==================================================" >> \$LOG_FILE
echo "Git Hash:   \$GIT_HASH" >> \$LOG_FILE
echo "Host Node:  \$HOST_NAME" >> \$LOG_FILE
echo "Env:        ${image}" >> \$LOG_FILE
echo "            flash_attn-2.8.3+rocm7.1_gfx90a_d128-cp310-linux_x86_64" >> \$LOG_FILE
echo "Run Script: Saved to run_script.sh (See file for exact config)" >> \$LOG_FILE
echo "==================================================" >> \$LOG_FILE
echo "" >> \$LOG_FILE
echo "--- Training Logs Start Below ---" >> \$LOG_FILE


nodes=( \$( scontrol show hostnames \$SLURM_JOB_NODELIST ) )
nodes_array=(\$nodes)
head_node=\${nodes_array[0]}
head_node_ip=$(hostname -i | awk '{print $1}')

echo "Master node: \$head_node"
echo "Master IP: \$head_node_ip"

export NCCL_SOCKET_IFNAME=hsn

module load singularity/4.1.0-slurm
export SINGULARITY_CACHEDIR="$MYSCRATCH/.singularity"
export SINGULARITYENV_HIP_VISIBLE_DEVICES=\$ROCR_VISIBLE_DEVICES
export SINGULARITYENV_NCCL_TIMEOUT=7200
export SINGULARITYENV_TORCH_NCCL_BLOCKING_WAIT=1
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

srun --ntasks=\$SLURM_JOB_NUM_NODES -c \$(( \$SLURM_GPUS_ON_NODE * 8 )) \
    singularity exec -B $MYSCRATCH/fakehome:${HOME} $MYSOFTWARE/singularity/rocm/${image}.sif \
    bash -c "
    export PATH=$MYSCRATCH/venvs/rocm-${project}/bin:\$PATH

    python -m torch.distributed.run --nproc_per_node=\$SLURM_GPUS_ON_NODE supervised-fine-tune.py \\
        \$CONFIG_FILE \\
        2>&1 | tee -a \$LOG_FILE
    "
EOF
        echo "Submitting Experiment: ${exp_name}"
        sbatch -p gpu "$script"
        rm "$script"
    done
done

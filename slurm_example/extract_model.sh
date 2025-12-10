#!/bin/bash

base_model="Qwen/Qwen3-4B"
# base_model="huggyllama/llama-7b"

peft_models=(
    /scratch/pawsey1151/alexchen5/LongLoRA/output/20251203_135521_Qwen3-4B_add_atom_action_qa_format
    /scratch/pawsey1151/alexchen5/LongLoRA/output/20251203_135521_Qwen3-4B_change_atom_action_qa_format
    /scratch/pawsey1151/alexchen5/LongLoRA/output/20251203_135522_Qwen3-4B_delete_below_atom_action_qa_format
    # /scratch/pawsey1151/alexchen5/LongLoRA/output/20251203_100415_llama-7b_train_data
)

checkpoints=(
    10 
    20 
    40 
    60
    80
)

project="LongLoRA"
image="pytorch_rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"

for peft_model in "${peft_models[@]}"; do
    for checkpoint in "${checkpoints[@]}"; do
        model_name="$(basename "$peft_model")_s${checkpoint}"
        script="tmp_extract_${model_name}.sh"
        cat > "$script" <<EOF
#!/bin/bash -l
#SBATCH --account=pawsey1151-gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --job-name=extract_${model_name}
#SBATCH --output=$MYSCRATCH/slurm/LongLoRA/extract_${model_name}-slurm-%j.out

cd \$MYSCRATCH/LongLoRA

module load singularity/4.1.0-slurm
export SINGULARITY_CACHEDIR="$MYSCRATCH/.singularity"
export SINGULARITYENV_HIP_VISIBLE_DEVICES=\$ROCR_VISIBLE_DEVICES
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

singularity exec -B $MYSCRATCH/fakehome:${HOME} $MYSOFTWARE/singularity/rocm/${image}.sif \
    bash -c "
    export PATH=$MYSCRATCH/venvs/rocm-${project}/bin:\$PATH

    mv ${peft_model}/checkpoint-${checkpoint}/adapter_model.safetensors ${peft_model}/checkpoint-${checkpoint}/adapter_model.bak.safetensors 

    python3 ${peft_model}/checkpoint-${checkpoint}/zero_to_fp32.py ${peft_model}/checkpoint-${checkpoint} ${peft_model}/checkpoint-${checkpoint}

    python3 get_trainable_weights.py --checkpoint_path ${peft_model}/checkpoint-${checkpoint} --trainable_params "embed,norm"

    python3 merge_lora_weights_and_save_hf_model.py \
        --base_model ${base_model} \
        --peft_model ${peft_model}/checkpoint-${checkpoint} \
        --context_size 16384 \
        --save_path \${MYSCRATCH}/LongLoRA/models/${model_name}
    "
EOF
        echo "Submitting Extract: ${model_name}"
        sbatch -p gpu "$script"
        rm "$script"
    done
done

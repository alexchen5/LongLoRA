#!/bin/bash

project="LongLoRA"
image="pytorch_rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"

script="tmp_checkmodel.sh"
cat > "$script" <<EOF
#!/bin/bash -l
#SBATCH --account=pawsey1151-gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --job-name=check_model
#SBATCH --output=$MYSCRATCH/slurm/LongLoRA/check_model-slurm-%j.out

cd \$MYSCRATCH/LongLoRA

module load singularity/4.1.0-slurm
export SINGULARITY_CACHEDIR="$MYSCRATCH/.singularity"
export SINGULARITYENV_HIP_VISIBLE_DEVICES=\$ROCR_VISIBLE_DEVICES
export SINGULARITYENV_PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

singularity exec -B $MYSCRATCH/fakehome:${HOME} $MYSOFTWARE/singularity/rocm/${image}.sif \
    bash -c "
    export PATH=$MYSCRATCH/venvs/rocm-${project}/bin:\$PATH

    python3 check_diff.py
    "
EOF
echo "Submitting Check"
sbatch -p gpu-dev "$script"
rm "$script"
#!/bin/bash -l
#SBATCH --account=<YOUR_PROJECT_ID>
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH --job-name=longlora_install
#SBATCH --output=<YOUR_PATH>/install-slurm-%j.out

project="LongLoRA"
image="pytorch_rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"

module load singularity/4.1.0-slurm
export SINGULARITY_CACHEDIR="$MYSCRATCH/.singularity"

cd $MYSCRATCH/flash-attention/

# 1. We prefix variables with SINGULARITYENV_ to pass inside the container
export SINGULARITYENV_HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
export SINGULARITYENV_GPU_ARCHS=gfx90a
export SINGULARITYENV_OPT_DIM="128"
export SINGULARITYENV_MAX_JOBS=8 
export SINGULARITYENV_FLASH_ATTENTION_FORCE_BUILD=TRUE

singularity exec -B $MYSCRATCH/fakehome:${HOME} $MYSOFTWARE/singularity/rocm/${image}.sif \
    bash -c "
    # 2. Activate Environment
    export PATH=$MYSCRATCH/venvs/rocm-${project}/bin:\$PATH

    # echo 'Cleaning previous build artifacts...'
    # rm -rf build/ 
    # rm -rf flash_attn.egg-info/
    
    # 3. Validation 
    echo \"---------------------------------------\"
    echo \"Build Configuration:\"
    echo \"Ninja Path:   \$(which ninja)\"
    echo \"Ninja Ver:    \$(ninja --version)\"
    echo \"OPT_DIM:      \$OPT_DIM\"
    echo \"MAX_JOBS:     \$MAX_JOBS\"
    echo \"---------------------------------------\"

    # 4. Run Install
    pip install . -v
    "

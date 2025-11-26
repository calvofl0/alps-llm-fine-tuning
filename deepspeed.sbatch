#!/bin/bash -l

# Job general details
#SBATCH --job-name test-deepspeed
#SBATCH --account <YOUR_ACCOUNT> 
#SBATCH --mail-user <YOUR_EMAIL>
#SBATCH --mail-type NONE
#SBATCH --time 00:29:59

# Paths and output
#SBATCH --output slurm-%j.out

# Ressources
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 4
#SBATCH --cpus-per-task 4

# Environment
#SBATCH --export NONE

# Load modules
module load cray/23.12
module load cray-python/3.11.5

export SLURM_EXPORT_ENV=ALL
export TRITON_CACHE_DIR=/dev/shm/$USER/triton
mkdir -p ${TRITON_CACHE_DIR}/autotune

MODEL_ID="./mistral-7B-Instruct-v0.3-hf"
OUTPUT_DIR="./results/mistral_sft_full"

EDF_FILE=./ngc-pytorch-deepspeed-24.06.toml
VIRTUAL_ENV=./.venv-ngc-pt-24.06
TRAIN_SCRIPT="./train_sft_mistral.py --model_id \"${MODEL_ID}\" --output_dir \"${OUTPUT_DIR}\""

echo $TRAIN_SCRIPT

MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)
MASTER_PORT=$(python -c "import socket;s=socket.socket(socket.AF_INET, socket.SOCK_STREAM);s.bind(('', 0));print(s.getsockname()[1]);s.close()")

HOSTFILE=hostfile
rm -f "${HOSTFILE}"

scontrol show hostnames ${SLURM_JOB_NODELIST} | while read host; do
    echo "${host} slots=${SLURM_NTASKS_PER_NODE}" >> $HOSTFILE
done

# Build venv
if [ ! -d ${VIRTUAL_ENV} ]; then
    srun -n 1 -N 1 --environment ${EDF_FILE} /bin/bash -c '\
        python3 -m venv --system-site-packages '"${VIRTUAL_ENV}"'; \
        '"${VIRTUAL_ENV}"'/bin/pip install --no-cache-dir -r requirements.txt'
fi

srun --environment ${EDF_FILE} /bin/bash -c '\
    '"${VIRTUAL_ENV}"'/bin/deepspeed \
        --no_ssh \
        --node_rank ${SLURM_NODEID} \
        --master_addr '"$MASTER_ADDR"' \
        --master_port '"$MASTER_PORT"' \
        --num_gpus ${SLURM_GPUS_ON_NODE} \
        --hostfile hostfile \
        --venv_script '"${VIRTUAL_ENV}"'/bin/activate \
        '"${TRAIN_SCRIPT}"''

echo THE END

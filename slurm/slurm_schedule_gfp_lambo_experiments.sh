#!/bin/bash
#SBATCH --job-name=GFP
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-596%15
#SBATCH --time=3-12:00:00

HOME_DIR=/home/pcq275/
COREL_DIR=${HOME_DIR}/corel/


CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh

## ENABLE CUDA ON Cluster
if [ $(cat ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh | wc -l) == 0 ]; then
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
fi

seed=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
batchsize=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
task=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})
n_start=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})
iterations=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $6}' ${CONFIG})
model=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $7}' ${CONFIG})

conda activate corel-env
echo "python ${COREL_DIR}/experiments/run_cold_warm_start_experiments_gfp_bo.py -b ${batchsize} -s ${seed} -p ${task} -n ${n_start} -m ${iterations} --model ${model}"
python ${COREL_DIR}/experiments/run_cold_warm_start_experiments_gfp_bo.py -b ${batchsize} -s ${seed} -p ${task} -n ${n_start} -m ${iterations} --model ${model}


exit 0
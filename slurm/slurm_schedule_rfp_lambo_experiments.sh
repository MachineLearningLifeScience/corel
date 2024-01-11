#!/bin/bash
#SBATCH --job-name=RFP
#SBATCH -p boomsma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-596
#SBATCH --time=3-12:00:00

HOME_DIR=/home/pcq275/
LAMBO_DIR=${HOME_DIR}/lambo/
COREL_DIR=${HOME_DIR}/corel/
OUTPUT_LOG=${COREL_DIR}experiments/slurm/rfp.log

NUM_ITERATIONS=100


CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh

## ENABLE CUDA ON Cluster
if [ $(cat ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh | wc -l) == 0 ]; then
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
fi

CONFIG=${COREL_DIR}slurm/rfp_runs.txt

algo=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
task=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
seed=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})
n_start=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})

if [ ${algo} == "lambo" ]; then
    conda activate poli__lambo
    echo "python ${LAMBO_DIR}/scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=${task} tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS} task.num_start_examples=${n_start}"
    python ${LAMBO_DIR}/scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=${task} tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS} task.num_start_examples=${n_start}
elif [ ${algo} == "corel" ]; then
    conda activate corel-env
    echo "python ${COREL_DIR}/experiments/run_cold_warm_start_experiments_rfp_bo.py -b 2 -s ${seed} -p ${task} -n ${n_start} -m ${NUM_ITERATIONS}"
    python ${COREL_DIR}/experiments/run_cold_warm_start_experiments_rfp_bo.py -b 2 -s ${seed} -p ${task} -n ${n_start} -m ${NUM_ITERATIONS}
else
    echo "Invalid algorithm ${algo}!"
    exit 1
fi

exit 0
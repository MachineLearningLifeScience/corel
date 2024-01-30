#!/bin/bash
#SBATCH --job-name=RFP_LAMBO
#SBATCH -p boomsma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --array=9-26
#SBATCH --time=3-12:00:00
#SBATCH --gres=gpu:1

HOME_DIR=/home/pcq275/
LAMBO_DIR=${HOME_DIR}/lambo/
COREL_DIR=${HOME_DIR}/corel/


CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/etc/profile.d/conda.sh

## ENABLE CUDA ON Cluster
if [ $(cat ${CONDA_PREFIX}/etc/conda/activate.d/env_vars.sh | wc -l) == 0 ]; then
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
fi

CONFIG=${COREL_DIR}slurm/rfp_lambo_config.txt

seed=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $2}' ${CONFIG})
batchsize=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $3}' ${CONFIG})
task=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $4}' ${CONFIG})
asset_dir=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $5}' ${CONFIG})
n_start=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $6}' ${CONFIG})
iterations=$(awk -v ArrayID=${SLURM_ARRAY_TASK_ID} '$1==ArrayID {print $7}' ${CONFIG})

conda activate poli__lambo
cd ${LAMBO_DIR}  # location unfortunately required for correct asset reference # TODO FIXME
echo "python ${LAMBO_DIR}/scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=${task} tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${iterations} task.num_start_examples=${n_start} task.batch_size=${batchsize} +task.data_path=${asset_dir}"
python ${LAMBO_DIR}/scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=${task} tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${iterations} task.num_start_examples=${n_start} task.batch_size=${batchsize} +task.data_path=${asset_dir}

exit 0
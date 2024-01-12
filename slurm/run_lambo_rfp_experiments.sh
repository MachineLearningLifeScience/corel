#!/bin/env bash
#SBATCH --job-name=lambo13
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=9-12:00:00

conda activate poli__lambo

NUM_ITERATIONS=100

COREL_DIR="/home/pcq275/corel/"
LAMBO_DIRECTORY="/home/pcq275/lambo/"

LOG_DIR="${COREL_DIR}/results/log/"


## Require task specifications on poli side to be included in hydra search path
# Lambo original proxy_rfp task:
cd ${LAMBO_DIRECTORY}
python ${LAMBO_DIRECTORY}/scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=proxy_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=2 optimizer.num_rounds=${NUM_ITERATIONS} | tee ${LOG_DIR}rfp_ref.log

# Lambo proxy_rfp task wrapped as poli task (same setup for benchmarking with the same backend)
for seed in 0 1 3 5 7 13; do
    python ${LAMBO_DIRECTORY}scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp_reference tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS} | tee ${LOG_DIR}poli_rfp_reference_${seed}.log
done


# Benchmark Poli Task: RFP FoldX stability and SASA
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=2 optimizer.num_rounds=${NUM_ITERATIONS} # 'hydra.searchpath=[file:///Users/rcml/poli/src/poli/objective_repository/foldx_rfp_lambo/]'

# FULL STABILITY SASA EXPERIMENT
for seed in 0 1 3 5 7 13; do
    # NOTE: num_gens implicitly trains representations
    # NOTE: weighted resampling scheme pushes performance - cannot be disabled.
    python ${LAMBO_DIRECTORY}scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS}
done

# LAMBO RUNNING FOLDX STABILITY SASA WARM (only PDBs available)
for seed in 0 1 3 5 7; do
    python ${LAMBO_DIRECTORY}scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS} task.num_start_examples=50
done

# Benchmark Poli Task RFP Foldx cold n=3 one reference protein
for seed in 0 1 3 5 7; do
    python ${LAMBO_DIRECTORY}scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp_cold tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS}
done
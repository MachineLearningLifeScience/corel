#!/bin/env bash
#SBATCH --job-name=lambo13
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=9-12:00:00

conda activate poli__lambo

NUM_ITERATIONS=64


## Require task specifications on poli side to be included in hydra search path
# Lambo original proxy_rfp task:
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=proxy_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=2 optimizer.num_rounds=${NUM_ITERATIONS}

# Lambo proxy_rfp task wrapped as poli task (same setup for benchmarking with the same backend)
for seed in 0 1 3 5 7 13; do
    python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp_reference tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS}
done


# Benchmark Poli Task: RFP FoldX stability and SASA
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=2 optimizer.num_rounds=${NUM_ITERATIONS} # 'hydra.searchpath=[file:///Users/rcml/poli/src/poli/objective_repository/foldx_rfp_lambo/]'

for seed in 0 1 3 5 7; do
    python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=${seed} optimizer.num_rounds=${NUM_ITERATIONS}
done
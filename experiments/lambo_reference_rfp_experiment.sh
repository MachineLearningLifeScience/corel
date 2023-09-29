#!/bin/env bash
#SBATCH --job-name=lambo13
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=9-12:00:00

conda activate poli__lambo

# for i in $(seq 0 9); do
#     python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp_internal tokenizer=protein surrogate=multi_task_exact_gp acquisition=ehvi trial_id=${i} optimizer.num_rounds=100
# done



# Require task specifications on poli side to be included in hydra search path
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=proxy_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=2 optimizer.num_rounds=100
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp_reference tokenizer=protein surrogate=multi_task_exact_gp acquisition=nehvi trial_id=13 optimizer.num_rounds=100


# Benchmark Poli Task
python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp tokenizer=protein surrogate=multi_task_exact_gp acquisition=ehvi trial_id=2 optimizer.num_rounds=100 # 'hydra.searchpath=[file:///Users/rcml/poli/src/poli/objective_repository/foldx_rfp_lambo/]'
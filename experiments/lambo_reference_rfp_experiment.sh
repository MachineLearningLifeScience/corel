#!/bin/env bash
conda activate poli__lambo

for i in $(seq 0 9); do
    python scripts/black_box_opt.py optimizer=lambo optimizer.encoder_obj=mlm task=poli_rfp_internal tokenizer=protein surrogate=multi_task_exact_gp acquisition=ehvi trial_id=${i}
done
conda activate corel-env

NUM_ITERATIONS=64

# run CoRel algorithm against lambo reference task
for seed in 0 1 5 6 13; do
    python experiments/run_cold_warm_start_experiments_rfp_bo.py -s ${seed} -b 16 -p foldx_rfp_lambo -m ${NUM_ITERATIONS}
done


# CoRel RFP known structures experiment 
for seed in 0 1 5 6 13; do
    python ./experiments/run_cold_warm_start_experiments_rfp_bo.py -b 16 -s ${seed} -p foldx_stability_and_sasa -d /home/pcq275/lambo/lambo/assets/foldx/ -m ${NUM_ITERATIONS}
done

# CoRel ice cold-start is foldx_stability with one RFP structure subselected
for seed in 0 1 5 6 13; do
    python ./experiments/run_cold_warm_start_experiments_rfp_bo.py -b 1 -s ${seed} -p foldx_stability_and_sasa -d /home/pcq275/lambo/lambo/assets/foldx/ -n 1 -m ${NUM_ITERATIONS}
done
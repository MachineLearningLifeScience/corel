conda activate corel-env

NUM_ITERATIONS=100

COREL_DIR="/home/pcq275/corel/"
LOG_DIR="${COREL_DIR}results/log/"

# run CoRel algorithm against lambo reference task
for seed in 0 1 3 5 7 13; do
    python ${COREL_DIR}experiments/run_cold_warm_start_experiments_rfp_bo.py -s ${seed} -b 2 -p foldx_rfp_lambo -m ${NUM_ITERATIONS} | tee ${LOG_DIR}poli_rfp_reference_COREL_${seed}.log
done


# CoRel RFP known structures experiment 
for seed in 0 1 3 5 7 13; do
    python ./experiments/run_cold_warm_start_experiments_rfp_bo.py -b 16 -s ${seed} -p foldx_stability_and_sasa -d /home/pcq275/lambo/lambo/assets/foldx/ -m ${NUM_ITERATIONS} | tee ${LOG_DIR}foldx_stability_and_sasa_COREL_${seed}.log
done

# CoRel warm start is foldx_stability with base RFP structures subselected
for seed in 0 1 3 5 7 13; do
    python ./experiments/run_cold_warm_start_experiments_rfp_bo.py -b 16 -s ${seed} -p foldx_stability_and_sasa -d /home/pcq275/lambo/lambo/assets/foldx/ -n 50 -m ${NUM_ITERATIONS} | tee ${LOG_DIR}foldx_stability_and_sasa_WARM_COREL_${seed}.log
done

# CoRel ice cold-start is foldx_stability with one RFP structure subselected
for seed in 0 1 3 5 7 13; do
    python ./experiments/run_cold_warm_start_experiments_rfp_bo.py -b 2 -s ${seed} -p foldx_stability_and_sasa -d /home/pcq275/lambo/lambo/assets/foldx/ -n 1 -m ${NUM_ITERATIONS} | tee ${LOG_DIR}foldx_stability_and_sasa_COLD_COREL_${seed}.log
done
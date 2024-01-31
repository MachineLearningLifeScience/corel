# run baselines for corel
COREL_DIR="/Users/sjt972/Projects/corel"
baselines=("RandomMutation")
problems=("gfp_cbas_elbo")
for seed in 0 1 3 5 7 13 23 42 71 123 29 37 73; do
    for problem in ${problems[@]}; do
        for baseline in ${baselines[@]}; do
            for starting_n in 3 16 50; do
                NUM_ITERATIONS=32
                echo "python ${COREL_DIR}/experiments/run_baselines_gfp.py \
                -s ${seed} \
                -b 16 \
                -p ${problem} \
                -a ${baseline} \
                -n ${starting_n} \
                -m ${NUM_ITERATIONS} 2>&1"
                # python ${COREL_DIR}/experiments/run_baselines_gfp.py \
                # -s ${seed} \
                # -b 16 \
                # -p ${problem} \
                # -a ${baseline} \
                # -n ${starting_n} \
                # -m ${NUM_ITERATIONS} 2>&1
            done
        done
    done
done


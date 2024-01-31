COREL_DIR="/Users/sjt972/Projects/corel"
baselines=("RandomMutation")
problems=("rfp_foldx_stability_and_sasa")
for seed in 1 3 5 7 13; do
    for problem in ${problems[@]}; do
        for baseline in ${baselines[@]}; do
            for starting_n in 6; do
                NUM_ITERATIONS=32
                echo "python ${COREL_DIR}/experiments/run_baselines_rfp.py \
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


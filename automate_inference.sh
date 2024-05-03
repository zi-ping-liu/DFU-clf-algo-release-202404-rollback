#!/bin/sh



# ########################################################################################################################################################
# # Experiment 1: Reproduce old baseline shiftwin model results (released in March)
# MODEL_PATH=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/reproduce_shiftwin/baseline
# HS=571
# BEST_EPOCH=14

# TRAIN_CSV=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/processed_WAUSI_BSV+shiftwin_20240304.csv
# TEST_CSV=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv

# MODEL_PRED_CV=${MODEL_PATH}/hs_${HS}/predictions_cv.csv

# ROOT_SAVE_DIR=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/release_summary/baseline_reproduce_hs_${HS}_${BEST_EPOCH}
# mkdir -p ${ROOT_SAVE_DIR}

# cp ${MODEL_PRED_CV} ${ROOT_SAVE_DIR}/predictions_cv.csv
# python3 stratified_analysis_v2.py ${TRAIN_CSV} ${MODEL_PRED_CV} false ${ROOT_SAVE_DIR}/predictions_cv_stats_include_sw.xlsx
# python3 stratified_analysis_v2.py ${TRAIN_CSV} ${MODEL_PRED_CV} true ${ROOT_SAVE_DIR}/predictions_cv_stats.xlsx
# for test_set in 0 1 2 3 4 5
# do
#     SAVE_PATH_INF=${ROOT_SAVE_DIR}/predictions_test_${test_set}.csv
#     python3 inference_v2.py ${TRAIN_CSV} ${TEST_CSV} ${test_set} ${MODEL_PATH}/hs_${HS} ${BEST_EPOCH} ${SAVE_PATH_INF}
#     python3 stratified_analysis_v2.py ${TEST_CSV} ${SAVE_PATH_INF} false ${ROOT_SAVE_DIR}/predictions_test_${test_set}_stats_include_sw.xlsx
#     python3 stratified_analysis_v2.py ${TEST_CSV} ${SAVE_PATH_INF} true ${ROOT_SAVE_DIR}/predictions_test_${test_set}_stats.xlsx
# done

########################################################################################################################################################
# Experiment 2: Retrain baseline shiftwin model on last-round training subjects but with updated data
MODEL_PATH=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/shiftwin_v5/baseline
HS=121
BEST_EPOCH=27

TRAIN_CSV=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/fixed20240502_processed_WAUSI_BSV+shiftwin_20240304_updated.csv
TEST_CSV=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/src/data/WAUSI_unifiedv6_BSV+slidingwindow_reverted_randsplit_5_20240430.csv

MODEL_PRED_CV=${MODEL_PATH}/hs_${HS}/predictions_cv.csv

ROOT_SAVE_DIR=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/results/release_summary/baseline_smallds_hs_${HS}_${BEST_EPOCH}
mkdir -p ${ROOT_SAVE_DIR}

cp ${MODEL_PRED_CV} ${ROOT_SAVE_DIR}/predictions_cv.csv
python3 stratified_analysis_v2.py ${TRAIN_CSV} ${MODEL_PRED_CV} false ${ROOT_SAVE_DIR}/predictions_cv_stats_include_sw.xlsx
python3 stratified_analysis_v2.py ${TRAIN_CSV} ${MODEL_PRED_CV} true ${ROOT_SAVE_DIR}/predictions_cv_stats.xlsx
for test_set in 0 1 2 3 4 5
do
    SAVE_PATH_INF=${ROOT_SAVE_DIR}/predictions_test_${test_set}.csv
    python3 inference_v2.py ${TRAIN_CSV} ${TEST_CSV} ${test_set} ${MODEL_PATH}/hs_${HS} ${BEST_EPOCH} ${SAVE_PATH_INF}
    python3 stratified_analysis_v2.py ${TEST_CSV} ${SAVE_PATH_INF} false ${ROOT_SAVE_DIR}/predictions_test_${test_set}_stats_include_sw.xlsx
    python3 stratified_analysis_v2.py ${TEST_CSV} ${SAVE_PATH_INF} true ${ROOT_SAVE_DIR}/predictions_test_${test_set}_stats.xlsx
done
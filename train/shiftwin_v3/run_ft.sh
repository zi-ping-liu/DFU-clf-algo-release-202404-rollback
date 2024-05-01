#!/bin/sh

GPU_DEVICE=$1 # assigned GPU device for current experiment
START=$2 # starting index of hyperparameter sets
END=$3 # ending index of hyperparameter sets
VERSION=$4 # version of unified csv used in training

# Define export path
EXPORT_DIR=/home/efs/ziping/workspaces/dfu/clf_algo_release_202404_rollback/out/shiftwin_v3/baseline_train_history_ft_${VERSION}
mkdir -p ${EXPORT_DIR}

# Loop to run the Python script from START to END
for i in $(seq $((START - 1)) $((END - 1)))
do
    python3 config_baseline.py $GPU_DEVICE ${i} ${VERSION} true > ${EXPORT_DIR}/hs_$((i+1))_out.txt 2>&1
done
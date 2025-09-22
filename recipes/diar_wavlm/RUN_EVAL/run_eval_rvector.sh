#!/bin/bash

DATASET=Compound
max_speakers=10
gpu_id=
infer_batchsize=32
init_rttm_type=  # eg: oracle
segmentation_duration=8
exp_name=
embedding_model=path/to/your/downloads/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin
embedding_name=pyannote-wespeaker-voxceleb-resnet34-LM
avg_ckpt_num=("10")
val_mode=("best")  #  "prev" "center")
val_metric=("Loss")  # ("Loss" "DER")
binarize_threshold=("0.50")
eval_set=("AMI" "AliMeeting" "AISHELL4" "NOTSOFAR-SC" "MSDWild" "RAMC" "DIHARD2" "DIHARD3" "CHiME6" "VoxConverse")
evaluate_on_dev=false
feature_extractor_type=

. ./utils/parse_options.sh

avg_ckpt_num_str=$(IFS=,; echo "${avg_ckpt_num[*]}")
val_mode_str=$(IFS=,; echo "${val_mode[*]}")
val_metric_str=$(IFS=,; echo "${val_metric[*]}")
binarize_threshold_str=$(IFS=,; echo "${binarize_threshold[*]}")
eval_set_str=$(IFS=,; echo "${eval_set[*]}")

conf_name=AED-TSVAD/${feature_extractor_type}/${exp_name}

echo "Running evaluation on datasets: $eval_set_str"
echo "CUDA_VISIBLE_DEVICES: $gpu_id"
echo "Configuration: $conf_name"
echo "Maximum speakers: $max_speakers"
echo "Seg duration: $segmentation_duration"

./eval_rvector.sh \
    $DATASET \
    $conf_name \
    $embedding_model \
    $embedding_name \
    $avg_ckpt_num_str \
    $val_mode_str \
    $val_metric_str \
    $binarize_threshold_str \
    $eval_set_str \
    $init_rttm_type \
    $gpu_id \
    $infer_batchsize \
    $max_speakers \
    $evaluate_on_dev \
    $segmentation_duration
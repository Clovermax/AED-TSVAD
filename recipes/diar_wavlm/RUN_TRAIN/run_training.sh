#!/usr/bin/env bash

stage=-1
stop_stage=-1
gpus=-1
port=1200

dataset=  # eg: Compound
conf_name=  # eg: model_wavlm-base+_rvector_cs-ft

project_folder=  # BASE_PATH

use_dual_opt=true  # joint train or not (in SSL-based frontend)

. ./utils/parse_options.sh

echo "stage: $stage"
echo "stop_stage: $stop_stage"

conf_dir=${project_folder}/recipes/diar_wavlm/configs/${dataset}
train_conf=${conf_dir}/${conf_name}.yaml

gpu_num=$(echo $gpus | awk -F',' '{print NF}')
echo "CUDA_VISIBLE_DEVICES: $gpus"
echo "gpu_num: $gpu_num"
echo "port: $port"

######################
# Training
######################

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 
    if (! $use_dual_opt); then
        echo "Stage1: use single-opt for model training..."
        CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
            --num_processes $gpu_num --main_process_port $port \
            ../run_single_opt.py -C $train_conf -M train
    else
        echo "Stage1: use dual-opt for model training..."
        CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
            --num_processes $gpu_num --main_process_port $port \
            ../run_dual_opt.py -C $train_conf -M train
    fi
fi

######################
# Resuming
######################

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    if (! $use_dual_opt); then
        echo "Restart: use single-opt for model training..."
        CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
            --num_processes $gpu_num --main_process_port $port \
            ../run_single_opt.py -C $train_conf -M train -R
    else
        echo "Restart: use dual-opt for model training..."
        CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
            --num_processes $gpu_num --main_process_port $port \
            ../run_dual_opt.py -C $train_conf -M train -R
    fi
fi

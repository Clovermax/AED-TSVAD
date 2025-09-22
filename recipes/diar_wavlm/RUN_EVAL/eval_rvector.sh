#!/usr/bin/env bash

# Licensed under the MIT license.
# Copyright 2024 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

set -eu
ulimit -n 2048

error_handler() {
    echo "Error occurred during inference or scoring. Skipping to next iteration..."
    set +e
}

trap 'error_handler' ERR

DATASET=$1
conf_name=$2
embedding_model=$3
embedding_name=$4
avg_ckpt_num=$5
val_mode=$6
val_metric=$7
binarize_threshold=$8
eval_set=$9
init_rttm_type=${10}
gpu_id=${11}
infer_batchsize=${12}
max_speakers=${13}
evaluate_on_dev=${14}
segmentation_duration=${15}

IFS=',' read -r -a avg_ckpt_num <<< "$avg_ckpt_num"
IFS=',' read -r -a val_mode <<< "$val_mode"
IFS=',' read -r -a val_metric <<< "$val_metric"
IFS=',' read -r -a binarize_threshold <<< "$binarize_threshold"
IFS=',' read -r -a eval_set <<< "$eval_set"

# general setup
stage=2
stop_stage=2
current_time=$(date +"%Y_%m_%d--%H-%M-%S")

YOUR_PATH=/home/nis/zeyan.song/projects/TSVAD  # substitute with your own path
recipe_root=$YOUR_PATH/recipes/diar_wavlm
storage_dir=/data/ssd1/zeyan.song/project_TSVAD/exp_data  # substitute with your own path

exp_root=$storage_dir/exp_${DATASET}

# inference setup
data_dir=$recipe_root/data/$DATASET

dset=dev
eset=eval
# oracle_rttm_dev=$data_dir/$dset/rttm
# oracle_rttm_eval=$data_dir/$eset/rttm

segmentation_step=0.1

# scoring setup
collar=0.00
REF_DIR=$data_dir
dscore_dir=$YOUR_PATH/dscore

diarization_dir=$exp_root/$conf_name
eval_output_dir=$diarization_dir/eval_output_max${max_speakers}/$embedding_name
mkdir -p $eval_output_dir

config_dir=`ls $diarization_dir/*.yaml | sort -r | head -n 1`

log_file=$eval_output_dir/${init_rttm_type}/eval_${current_time}.log
mkdir -p $eval_output_dir/${init_rttm_type}
if [ -f $log_file ]; then
    rm $log_file
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 

    train_log=`du -h $diarization_dir/*.log | sort -rh | head -n 1 | awk '{print $NF}'`
    cat $train_log | grep 'Loss/DER' | awk -F ']:' '{print $NF}' > $diarization_dir/val_metric_summary.lst

    for vmode in "${val_mode[@]}"; do
        for vmetric in "${val_metric[@]}"; do
            for acn in "${avg_ckpt_num[@]}"; do
                for bt in "${binarize_threshold[@]}"; do
                    echo "Stage2: model inference..."
                    echo "VAL_MODE=$vmode, VAL_METRIC=$vmetric, AVG_CPKT_NUM=$acn, BINARIZE_THRESHOLD=$bt"
                    infer_affix=_segmentation_step_${segmentation_step}_binarize_threshold_${bt}

                    echo -e "\n****************************************************************\n" >> $log_file
                    echo -e "In iter: VAL_MODE=$vmode, VAL_METRIC=$vmetric, AVG_CPKT_NUM=$acn, BINARIZE_THRESHOLD=$bt: \n" >> $log_file

                    # evaluate on dev set
                    if [ "$evaluate_on_dev" = "true" ]; then
                        CUDA_VISIBLE_DEVICES=$gpu_id python ${recipe_root}/infer_avg_rvector.py \
                            --configuration $config_dir \
                            --in_wav_scp ${data_dir}/${dset}/wav.scp \
                            --out_dir ${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${dset}/sessions \
                            --uem_file ${data_dir}/${dset}/all.uem \
                            --init_rttms ${data_dir}/${dset}/init_rttms/${init_rttm_type} \
                            --embedding_model $embedding_model \
                            --avg_ckpt_num $acn \
                            --val_metric $vmetric \
                            --val_mode $vmode \
                            --val_metric_summary $diarization_dir/val_metric_summary.lst \
                            --segmentation_duration $segmentation_duration \
                            --segmentation_step $segmentation_step \
                            --batch_size $infer_batchsize \
                            --max_speakers $max_speakers \
                            --training_embedding_root ${data_dir}/train/${DATASET}/embs_single_spk \
                            --apply_median_filtering \
                            --binarize_threshold $bt \

                        oracle_rttm_dev=$data_dir/$dset/rttm
                        echo "Scoring..."
                        SYS_DIR=${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}
                        OUT_DIR=${SYS_DIR}/${dset}/sessions
                        python ${dscore_dir}/score.py \
                            -r ${REF_DIR}/${dset}/rttm \
                            -s $OUT_DIR/*.rttm --collar ${collar} \
                            > $OUT_DIR/../result_collar${collar}

                        hyp_rttm_dev=$eval_output_dir/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${dset}/hyp_${dset}.rttm
                        cat $eval_output_dir/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${dset}/sessions/*.rttm > $hyp_rttm_dev

                        echo -e "Evaluation on $dset: \n" >> $log_file

                        if [ -f $hyp_rttm_dev ]; then
                            local/analysis_diarization.sh 0.00 $eval_output_dir ${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${dset} DER_result_collar0.00 $oracle_rttm_dev $hyp_rttm_dev ${data_dir}/${dset}/all.uem | grep ALL
                            local/analysis_diarization.sh 0.25 $eval_output_dir ${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${dset} DER_result_collar0.25 $oracle_rttm_dev $hyp_rttm_dev ${data_dir}/${dset}/all.uem | grep ALL
                        fi
                        echo "DER with collar 0.00:" >> $log_file
                        tail -n 1 ${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${dset}/DER_result_collar0.00/temp.details >> $log_file
                        echo "DER with collar 0.25:" >> $log_file
                        tail -n 1 ${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${dset}/DER_result_collar0.25/temp.details >> $log_file
                    fi

                    # evaluate on individual eval set
                    for esubset in "${eval_set[@]}"; do
                        CUDA_VISIBLE_DEVICES=$gpu_id python ${recipe_root}/infer_avg_rvector.py \
                            --configuration $config_dir \
                            --in_wav_scp ${data_dir}/${eset}/$esubset/wav.scp \
                            --out_dir ${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${eset}/$esubset/sessions \
                            --uem_file ${data_dir}/${eset}/$esubset/all.uem \
                            --init_rttms ${data_dir}/${eset}/$esubset/init_rttms/${init_rttm_type} \
                            --embedding_model $embedding_model \
                            --avg_ckpt_num $acn \
                            --val_metric $vmetric \
                            --val_mode $vmode \
                            --val_metric_summary $diarization_dir/val_metric_summary.lst \
                            --segmentation_duration $segmentation_duration \
                            --segmentation_step $segmentation_step \
                            --batch_size $infer_batchsize \
                            --max_speakers $max_speakers \
                            --training_embedding_root ${data_dir}/train/${DATASET}/embs_single_spk \
                            --apply_median_filtering \
                            --binarize_threshold $bt \

                        oracle_rttm_eval=$data_dir/$eset/$esubset/rttm
                        echo "Scoring..."
                        SYS_DIR=${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}
                        OUT_DIR=${SYS_DIR}/${eset}/$esubset/sessions
                        python ${dscore_dir}/score.py \
                            -r ${REF_DIR}/${eset}/$esubset/rttm \
                            -s $OUT_DIR/*.rttm --collar ${collar} \
                            > $OUT_DIR/../result_collar${collar}

                        hyp_rttm_eval=$eval_output_dir/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${eset}/$esubset/hyp_${eset}.rttm
                        cat $eval_output_dir/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${eset}/$esubset/sessions/*.rttm > $hyp_rttm_eval

                        echo -e "\nEvaluation on $eset $esubset: \n" >> $log_file

                        if [ -f $hyp_rttm_eval ]; then
                            local/analysis_diarization.sh 0.00 $eval_output_dir ${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${eset}/$esubset DER_result_collar0.00 $oracle_rttm_eval $hyp_rttm_eval ${data_dir}/${eset}/$esubset/all.uem | grep ALL
                            local/analysis_diarization.sh 0.25 $eval_output_dir ${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${eset}/$esubset DER_result_collar0.25 $oracle_rttm_eval $hyp_rttm_eval ${data_dir}/${eset}/$esubset/all.uem | grep ALL                            
                        fi
                        echo "DER with collar 0.00:" >> $log_file
                        tail -n 1 ${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${eset}/$esubset/DER_result_collar0.00/temp.details >> $log_file
                        echo "DER with collar 0.25:" >> $log_file
                        tail -n 1 ${eval_output_dir}/${init_rttm_type}/infer$infer_affix/metric_${vmetric}_${vmode}/avg_ckpt${acn}/${eset}/$esubset/DER_result_collar0.25/temp.details >> $log_file
                    done
                    echo -e "\n****************************************************************\n" >> $log_file
                done
            done
        done
    done

fi

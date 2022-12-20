#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


preprocessed_dataset_path="preproc_data_train/"
current_exp_name="ood-dataset-train-30august/"


function cli(){
    echo "predict_after_train_another_dt"
    preprocessed_dataset_path=${1}
    exp_name=${2}
    trained_checkpoint_path=${3}
    device=${5}
    split=${6}


    output_dir="experiments/${exp_name}/${preprocessed_dataset_path}"
    python3 -m Bart_Program.predict --input_dir "${output_dir}" --save_dir "${output_dir}/log_predict_after_train_another_dt" --ckpt "${trained_checkpoint_path}" --device "${device}" --test_file "${split}.pt" --batch_size 16 --cli
}



cli ${preprocessed_dataset_path} ${current_exp_name} "experiments/${current_exp_name}/checkpoint/$(ls -1rt experiments/${current_exp_name}/checkpoint/  | tail -n1)" "cuda:5" "val"

#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


function predict_using_cached_preprocessing(){
    echo "predict_after_train_another_dt"
    preprocessed_dataset_path=${1}
    exp_name=${2}
    trained_checkpoint_path=${3}
    new_dataset=${4}
    device=${5}
    split=${6}
    original_model=${7}

    output_dir="experiments/${exp_name}/${preprocessed_dataset_path}"

    python3 -m Bart_Program.predict --input_dir "${output_dir}" --save_dir "${output_dir}/log_predict_after_train_another_dt" --ckpt "${trained_checkpoint_path}" --device "${device}" --test_file "${split}.pt" --batch_size 64 #16
}


function predict_with_preprocessing(){
    echo "predict_after_train_another_dt"
    preprocessed_dataset_path=${1}
    exp_name=${2}
    trained_checkpoint_path=${3}
    new_dataset=${4}
    device=${5}
    split=${6}
    original_model=${7}


    cp "${trained_checkpoint_path}"/model/* "${trained_checkpoint_path}"/
    cp "${trained_checkpoint_path}"/tokenizer/* "${trained_checkpoint_path}"/

    output_dir="experiments/${exp_name}/${preprocessed_dataset_path}"

    python3 -m Bart_Program.preprocess --input_dir "${new_dataset}" --output_dir "${output_dir}" --model_name_or_path "${original_model}"
    cp data/kb.json "${output_dir}"

    python3 -m Bart_Program.predict --input_dir "${output_dir}" --save_dir "${output_dir}/log_predict_after_train_another_dt" --ckpt "${trained_checkpoint_path}" --device "${device}" --test_file "${split}.pt" --batch_size 64
}


preprocessed_dataset_path="preproc_data_train/"
original_bart="KQAPro_ckpt/original_bart_weights/bart-base/"
current_exp_name="in_ontology_kqa-pro/"
hdd_dataset="HallucinationSimulationFramework/out_folder/hdd-dataset"
device="cuda:5"
predict_with_preprocessing ${preprocessed_dataset_path} ${current_exp_name} "experiments/${current_exp_name}/checkpoint/$(ls -1rt experiments/${current_exp_name}/checkpoint/  | tail -n1)" "${hdd_dataset}" "${device}" "val" "${original_bart}"

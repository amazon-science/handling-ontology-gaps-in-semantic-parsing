#!/usr/bin/env bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


function train(){
    echo "train_and_predict"
    preprocessed_dataset_path=${1}
    downloaded_checkpoints_path=${2}
    exp_name=${3}
    epochs=${4}
    device=${5}
    dataset_path=${6}
    exp_folder="experiments/${exp_name}/"
    mkdir -p "${exp_folder}"
    #python3 -m Bart_Program.preprocess --input_dir "${dataset_path}" --output_dir "${exp_folder}/${preprocessed_dataset_path}" --model_name_or_path "${downloaded_checkpoints_path}"
    cp data/kb.json "${exp_folder}/${preprocessed_dataset_path}"
    python3 -m Bart_Program.train --input_dir "${exp_folder}/${preprocessed_dataset_path}" --output_dir "${exp_folder}/checkpoint/" --save_dir "${exp_folder}/log_train/" --model_name_or_path "${exp_folder}/${preprocessed_dataset_path}" --num_train_epochs "${epochs}" --device "${device}"

}


preprocessed_dataset_path="preproc_data_train/"
original_bart="KQAPro_ckpt/original_bart_weights/bart-base/"  # download it from https://github.com/shijx12/KQAPro_Baselines/tree/master/Bart_Program#checkpoints
current_exp_name="in_ontology_kqa-pro/"  # do not write "checkpoint" in the current_exp_name
nsp_in_ontology_dataset="HallucinationSimulationFramework/out_folder/NSP_in_ontology_dataset/"
epochs=3

train "${preprocessed_dataset_path}" "${original_bart}" "${current_exp_name}" "${epochs}" "cuda:6" "${nsp_in_ontology_dataset}"
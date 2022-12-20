#!/usr/bin/env bash
#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: CC-BY-NC-4.0



#this script assumes that the user has already downloaded the KQA-PRO and TOP v2 datasets into the folders (data/ and data/TOPv2_Dataset/)
# create new split from KQA-PRO
python3 split_kqa-pro_dataset.py --input_folder data/KQAPro.IID/ --output_folder data_split_60
# generate the NSP in-ontology dataset
python3 generate_NSP_in_ontology_dataset.py --input_folder data_split_60
# generate the HDD dataset
python3 generate_Hallucination_Detection_Dataset.py --input_folder data_split_60
python3 convert_hdd_to_autodetect.py --input_hdd "out_folder/hdd-dataset/" --output_hdd_autodetect "out_folder/hdd-dataset-autodetect/"
# generate the OOD dataset
python3 generate_OOD_dataset.py --input_folder_KQA data_split_60 --input_folder_TOPv2 "data/TOPv2_Dataset/"
python3 generate_OOD_dataset.py --input_folder_KQA data_split_60 --input_folder_TOPv2 "data/TOPv2_Dataset/" --output_folder "out_folder/ood-dataset_autodetect/" --autodetect



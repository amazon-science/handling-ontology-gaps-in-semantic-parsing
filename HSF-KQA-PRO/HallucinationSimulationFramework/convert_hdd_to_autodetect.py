#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""


import argparse
import os
from typing import *
from tqdm import tqdm
from utilities import read_jsonl, write_jsonl




def convert_dataset_to_autodetect(dataset: List[dict], print_stats:bool=False):
    missing_ontology_placeholder: List[dict] = [{"function": "<missing_ontology>", "dependencies": [], "inputs": []}]
    out_of_ontology_counter, in_ontology_counter = 0, 0
    for line in dataset:
        if line["is_hallucinations"] == 1:
            line["program"] = missing_ontology_placeholder
            out_of_ontology_counter += 1
        else:
            line["is_hallucinations"] = 0
            in_ontology_counter += 1
    if print_stats: print("out-of-ontology", out_of_ontology_counter, "in-ontology", in_ontology_counter)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_hdd', type=str, help='path to HDD folder (default "out_folder/hdd-dataset")',
                        default="out_folder/hdd-dataset")
    parser.add_argument('--output_hdd_autodetect', type=str,
                        help='destination folder for the autodetect hdd (default data/kb.json)',
                        default="out_folder/autodetect-hdd-dataset")
    args = parser.parse_args()

    os.makedirs(args.output_hdd_autodetect, exist_ok=True)

    for split in tqdm(["train", "val", "test"], desc="Converting HDD into Autodetect HDD ..."):

        dataset: List[dict] = read_jsonl(f"{args.input_hdd}/{split}.json")[0]
        dataset = convert_dataset_to_autodetect(dataset)
        write_jsonl(f"{args.output_hdd_autodetect}/{split}.json", dataset)

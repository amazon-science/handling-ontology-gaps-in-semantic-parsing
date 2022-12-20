#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

import argparse
import os

from typing import *


from ontology_gap_utilities import disrupt_knowledge_

from utilities import read_jsonl, write_jsonl, read_json


def generate_in_ontology_dataset(ontology_to_drop: List[str], input_folder: str, dest_folder: str) -> None:
    """
    This function creates the NSP in-ontology dataset used to train the KQA-PRO BART.
    ontology_to_drop: List[str] = Contains all the ontologies keyword that we want remove.
    """
    os.makedirs(dest_folder, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"#===== {split} =====#")
        dataset_jsonl = read_jsonl(f"{input_folder}/{split}.json")[0]
        print("original dataset len:", len(dataset_jsonl))

        new_dataset = disrupt_knowledge_(
            dataset_jsonl,
            ontology_to_drop,
        )
        print("NSP in ontology len:", len(new_dataset))

        write_jsonl(f"{dest_folder}/{split}.json", new_dataset)
        read_jsonl(f"{dest_folder}/{split}.json")  # we read it again, only to check if the dump was successful.



if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='path to the input folder, it assume the presence of train.jsonl, val.jsonl (default data_split/)', default="data_split_60/")
    parser.add_argument('--output_folder', type=str, help='output path with the new splits (default out_folder/in-ontology_dataset)', default="out_folder/nsp-in_ontology/")
    parser.add_argument('--ontology_split_json', type=str, help='', default="ontology_split.json")
    args = parser.parse_args()

    ontology_split = read_json("ontology_split.json")
    ontology_to_drop = ontology_split["train"]
    ontology_dev_specific = ontology_split["dev"]
    ontology_test_specific = ontology_split["test"]


    generate_in_ontology_dataset(
        ontology_to_drop + ontology_dev_specific + ontology_test_specific, input_folder=args.input_folder, dest_folder=args.output_folder
    )

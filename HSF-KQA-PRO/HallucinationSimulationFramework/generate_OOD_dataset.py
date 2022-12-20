#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
import argparse
import os
from typing import *

from utilities import read_jsonl, write_jsonl, read_top_dataset, read_json

from ontology_gap_utilities import disrupt_knowledge_

def add_in_ontology_marker(new_dataset: List[dict]) -> List[dict]:
    for i, example in enumerate(new_dataset):
        if "is_hallucinations" not in example:
            new_dataset[i]["is_hallucinations"] = 0
    return new_dataset


def generate_ood_dataset(ontology_to_drop: List[str], input_folder_kqa: str, input_folder_topv2: str, output_folder: str, autodetect: bool=False) -> None:


    os.makedirs(output_folder, exist_ok=True)

    for split in ["train", "val", "test"]:
        print(f"#===== {split} =====#")
        dataset_jsonl = read_jsonl(f"{input_folder_kqa}/{split}.json")[0]


        if split in ["train", "val"]:
            # not used. It is requested by KQA-PRO processor.
            write_jsonl(f"{output_folder}/{split}.json", dataset_jsonl)
            continue

        new_dataset = disrupt_knowledge_(
            dataset_jsonl,
            ontology_to_drop,
            replace_with_unk=autodetect,
            keep_ooo_questions_without_mrl=False
            )  # we collect only in-ontology sentences

        new_dataset = add_in_ontology_marker(new_dataset)

        print(f"dataset split {split} - initial len {len(new_dataset)}")
        
        fb_data: List[dict] = read_top_dataset(top_v2_folder_path=input_folder_topv2, autodetect=autodetect, only_test_split=True)
        # we read the top_dataset v2

        print(f"Len fb data v2: {len(fb_data)}")

        new_dataset.extend(fb_data)  # we merge NSP in-ontology and OOD sentences (top v2 dataset)

        print(f"ood dataset split {split} - final len {len(new_dataset)}")

        write_jsonl(f"{output_folder}/{split}.json", new_dataset)





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder_KQA', type=str, help='path to the input folder, it assume the presence of train.jsonl, val.jsonl (default data_split)', default="data_split_60_3/")
    parser.add_argument('--input_folder_TOPv2', type=str, help='path to the input folder, it assume the presence of train.jsonl, val.jsonl (default data/TOPv2_Dataset/)', default="data/TOPv2_Dataset/")
    parser.add_argument('--output_folder', type=str, help='output path with the new splits (default out_folder/ood-dataset)', default="out_folder/ood-dataset/")
    parser.add_argument('--autodetect', action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()

    ontology_split = read_json("ontology_split.json")
    ontology_to_drop = ontology_split["train"]
    ontology_dev_specific = ontology_split["dev"]
    ontology_test_specific = ontology_split["test"]

    full_out_of_ontology: List[str] = ontology_to_drop + ontology_dev_specific + ontology_test_specific

    generate_ood_dataset(full_out_of_ontology, args.input_folder_KQA, args.input_folder_TOPv2, args.output_folder, autodetect=args.autodetect)

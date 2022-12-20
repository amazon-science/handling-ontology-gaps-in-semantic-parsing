#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

from collections import defaultdict
import os
import json

from typing import *
from generate_Hallucination_Detection_Dataset import collect_in_ontology_question, collect_out_of_ontology_question


from ontology_gap_utilities import disrupt_knowledge_

from utilities import read_jsonl, write_jsonl, read_json


def autodetect_remove_program(dataset: List[dict]):
    for line in dataset:
        line["program"] = [
                {"function": "<missing_ontology>", "dependencies": [], "inputs": []}
            ]  # we replace the program seq with an empty program seq, for the selected ids
        line["is_hallucinations"] = 1
    
    return dataset


def bart_autodetect_hallucination_generate_dataset(ontology_train: List[str], ontology_dev: List[str], ontology_test: List[str]) -> None:
    destination_folder: str = "out_folder/bart_autodetect_dataset_train_dev/"
    os.makedirs(destination_folder, exist_ok=True)

    expands: Dict[list] = defaultdict(list)

    ontology_split_based: Dict[str, List[str]] = {"train": ontology_train, "val": ontology_dev, "test": ontology_test}

    for split in ["train", "val", "test"]:
        dataset_jsonl = read_jsonl(f"data_split_60/{split}.json")[0]
        for alternative_split in ["train", "val", "test"]:
            if alternative_split == split: continue
            second_alternative_split = list({"train", "val", "test"} - {split, alternative_split})[0]
            ontology_to_avoid = ontology_split_based[split] + ontology_split_based[second_alternative_split]
            ooo_questions_expands = collect_out_of_ontology_question(dataset_jsonl, ontology_split_based[alternative_split], ontology_to_avoid, keep_mrl=True, autodetect=True)
            ooo_questions_expands = autodetect_remove_program(ooo_questions_expands)
            expands[alternative_split].extend(ooo_questions_expands)

    for split in ["train", "val", "test"]:
        print(f"#===== {split} =====#")
        dataset_jsonl = read_jsonl(f"data_split_60/{split}.json")[0]
        print("dataset len", len(dataset_jsonl))
        if split == "train":
            to_drop = ontology_dev + ontology_test
            to_keep = ontology_train
        elif split == "val":
            to_drop = ontology_train + ontology_test
            to_keep = ontology_dev
        else:
            to_drop = ontology_train + ontology_dev
            to_keep = ontology_test

        in_domain: List[Dict] = collect_in_ontology_question(dataset_jsonl, to_drop + to_keep)
        
        out_of_domain = collect_out_of_ontology_question(dataset_jsonl, to_keep, to_drop, keep_mrl=True, autodetect=True)
        out_of_domain = autodetect_remove_program(out_of_domain)

        in_domain.extend(expands[split])
        in_domain.extend(out_of_domain)
        new_dataset = in_domain


        counter, total = 0, 0
        for idx, entry in enumerate(new_dataset):
            total += 1
            if "is_hallucinations" not in entry:
                new_dataset[idx]["is_hallucinations"] = 0
            if new_dataset[idx]["is_hallucinations"] == 0:
                counter += 1

            
        print("in ontology ", counter)
        print("total ", total)
        print("out-of-ontology ", total-counter)

        print("final preproc len: ", len(new_dataset))

        write_jsonl(f"{destination_folder}/{split}.json", new_dataset)
        read_jsonl(f"{destination_folder}/{split}.json")



if __name__ == "__main__":
    ...
    
    ontology_split = read_json("ontology_split.json")
    ontology_train = ontology_split["train"]
    ontology_dev = ontology_split["dev"]
    ontology_test = ontology_split["test"]

    bart_autodetect_hallucination_generate_dataset(ontology_train=ontology_train, ontology_dev=ontology_dev, ontology_test=ontology_test)


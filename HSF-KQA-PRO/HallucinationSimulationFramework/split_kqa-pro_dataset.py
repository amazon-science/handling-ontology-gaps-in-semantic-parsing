#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
import os
import math
from typing import *
import random
import argparse

from utilities import read_jsonl, write_jsonl


class SanityCheckFailed(Exception):
    """SanityCheckFailed exception"""
    pass




def assign_id_to_each_samples(dataset: List[dict]) -> List[dict]:

    for id_sample in range(len(dataset)):
        dataset[id_sample]["id_sample"] = id_sample
    return dataset


def compute_percentage(total: int, percent: int) -> int:
    assert 0 < percent < 100, f"ERROR: compute_percentage: percent is {percent} and is out of [0, 100] range"
    return math.floor((total / 100) * percent)

def dataset_sanity_check(train_split: List[dict], val_split: List[dict], test_split: List[dict]) -> None:
    '''
    This sanity check will ensure that each id_sample occurs in a single dataset split.
    '''
    id_seen = set()
    for dataset in [train_split, val_split, test_split]:
        for sample in dataset:
            if sample["id_sample"] in id_seen:
                raise SanityCheckFailed
            id_seen.add(sample["id_sample"])
            
def split_dataset(dataset_folder: str, output_folder: str, percentage: int) -> None:
    train_dataset: List[str] = read_jsonl(os.path.join(dataset_folder, "train.json"))[0]
    val_dataset: List[str] = read_jsonl(os.path.join(dataset_folder, "val.json"))[0]
    print("Original train size: ", len(train_dataset))
    print("Original val size: ", len(val_dataset))

    merged_dataset: List[str] = train_dataset + val_dataset  # We merge the train and val set

    merged_dataset = assign_id_to_each_samples(merged_dataset)

    random.shuffle(merged_dataset)  # We shuffle the merged dataset

    total_len: int = len(merged_dataset)

    print("Total size: ", total_len)

    val_and_test_size: int = compute_percentage(total_len, percentage)
    print(val_and_test_size)

    val_split: List[str] = merged_dataset[:val_and_test_size]

    test_split: List[str] = merged_dataset[val_and_test_size: val_and_test_size + val_and_test_size]
    train_split: List[str] = merged_dataset[val_and_test_size + val_and_test_size:]
    try:
        dataset_sanity_check(train_split, val_split, test_split)
    except SanityCheckFailed:
        print("Sanity check failed, id_samples are shared in multiple splits")
        exit(-1)

    print(f"val.json lenght: {len(val_split)}")
    print(f"test.json lenght: {len(test_split)}")
    print(f"train.json lenght: {len(train_split)}")

    write_jsonl(os.path.join(output_folder, "val.json"), val_split)
    write_jsonl(os.path.join(output_folder, "test.json"), test_split)
    write_jsonl(os.path.join(output_folder, "train.json"), train_split)




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='insert an integer to fix the randomness (default 42)', default=42)
    parser.add_argument('--percentage', type=int, help='percentage for dev and test split (default is 20%)', default=20)
    parser.add_argument('--input_folder', type=str, help='path to the input folder, it assume the presence of train.jsonl, val.jsonl (default data/)', default="data/")
    parser.add_argument('--output_folder', type=str, help='output path with the new splits (default data/split_60)', default="data_split_60_3/")


    args = parser.parse_args()

    SEED: int = args.seed
    PERCENTAGE: int = args.percentage  # percentage of each split of dev and test.
    random.seed(SEED)  # Randomness fixed
    os.makedirs(args.output_folder, exist_ok=True)
    split_dataset(args.input_folder, args.output_folder, PERCENTAGE)


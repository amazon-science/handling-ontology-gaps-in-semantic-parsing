#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

import json
import glob
from typing import *
from dataclasses import dataclass
from itertools import chain


import pandas as pd


@dataclass
class KQASample:
    question: str
    choices: List[str]
    program: List[dict]
    sparql: str
    answer: str
    is_hallucinations: int

def read_json(filename: str) -> dict:
    assert filename.endswith(".json"), "read_json Error the file is not a .json file"
    with open(filename, "r") as reader:
        return json.load(reader)


def write_json(filename: str, dictionary: dict) -> None:
    assert filename.endswith(".json"), "write_json() Error: the file is not a .json file"
    with open(filename, "w") as writer:
        json.dump(obj=dictionary, fp=writer, indent=4, sort_keys=False)


def read_jsonl(filename: str) -> List[dict]:
    assert ".json" in filename, "read_jsonl() Error: the file is not a jsonl file"
    with open(filename, "r") as reader:
        return [json.loads(line.strip()) for line in reader.readlines()]


def write_jsonl(filename: str, dictionary_list: List[dict]) -> None:
    assert ".json" in filename, "write_jsonl() Error: the file is not a jsonl file"
    total_len = len(dictionary_list)
    with open(filename, "w") as writer:
        writer.write("[")
        for idx, line in enumerate(dictionary_list):
            json.dump(line, writer)
            if idx != total_len - 1:
                writer.write(",")
        writer.write("]")


def read_top_dataset(top_v2_folder_path: str, autodetect: bool = False, only_test_split: bool=True) -> List[dict]:
    split: str = "*test" if only_test_split else "*"
    dataset_list: List[List[str]] = [pd.read_csv(filename, sep='\t').iloc[:, 1].tolist() for filename in glob.glob(f"{top_v2_folder_path}/{split}.tsv")]
    intent_list: List[str] = list(set(chain.from_iterable(dataset_list)))
    program = [] if not autodetect else [{"function": "<missing_ontology>", "dependencies": [], "inputs": []}]   # we replace the program seq with an empty program seq, for the selected ids 
    choices = ["Joe Pantoliano", "James E. Reilly", "Max Fleischer", "Rahul Dev Burman", "Julia Ormond", "Simon Cowell", "William Henry Harrison", "Sylvester Stallone", "Richard Gere", "Paul Simon"]
    res = [KQASample(intent, choices=choices, program=program, sparql="", answer="Sylvester Stallone", is_hallucinations=1).__dict__ for intent in intent_list]
    return res





if __name__ == "__main__":
    print(read_json("data/kb.json").keys())

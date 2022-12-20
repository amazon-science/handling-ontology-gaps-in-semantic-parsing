#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
import argparse
import json
import random
from collections import defaultdict
from typing import *


def read_json(filename) -> dict:
    with open(filename, "r") as reader:
        return json.load(reader)


def write_json(filename, dictionary) -> dict:
    with open(filename, "w") as writer:
        json.dump(dictionary, writer, indent=4, sort_keys=True)


def build_relations_attributes(kb: dict) -> Tuple[dict, dict]:
    relations = defaultdict(lambda: defaultdict(int))
    attributes = defaultdict(lambda: defaultdict(int))

    for q_id, instance in kb["entities"].items():
        for attrib in instance["attributes"]:
            attribute = attrib["key"]
            attributes[attribute]["train"], attributes[attribute]["val"], attributes[attribute]["test"] = 0, 0, 0

        for rel in instance["relations"]:
            relantions_list = list(set(rel["qualifiers"].keys()))
            for r in relantions_list:
                relations[r]["train"], relations[r]["val"], relations[r]["test"] = 0, 0, 0

    return relations, attributes


def get_properties_correlations(relations: dict, attributes: dict, kqa_input_path: str) -> Tuple[dict, dict]:
    for split in ["train", "val", "test"]:

        dataset = read_json(f"{kqa_input_path}/{split}.json")

        for line in dataset:
            for function in line["program"]:
                for inp in function["inputs"]:
                    if inp in relations:
                        relations[inp][split] += 1
                    if inp in attributes:
                        attributes[inp][split] += 1

    return relations, attributes

def clean_dict(dictionary):
    to_delete = []
    for name, frequency in dictionary.items():
        if frequency["train"] == frequency["test"] == frequency["val"] == 0:
            to_delete.append(name)

    for item_to_delete in to_delete:
        del dictionary[item_to_delete]
    return dictionary


def build_relations(kb_path: str, kqa_input_path: str, output_path_relations: str):
    kb: dict = read_json(kb_path)
    relations, attributes = build_relations_attributes(kb)
    relations, _ = get_properties_correlations(relations, attributes, kqa_input_path)
    relations = clean_dict(relations)
    write_json(output_path_relations, relations)
    return relations

def present_only(relations_dict):
    train_only, val_only, test_only = set(), set(), set()

    for rel, item in relations_dict.items():
        count_train, count_val, count_test = item["train"], item["val"], item["test"]
        if count_train == 0 and count_val == 0 and count_test != 0:
            test_only.add(rel)
        elif count_train == 0 and count_val != 0 and count_test == 0:
            val_only.add(rel)
        elif count_train != 0 and count_val == 0 and count_test == 0:
            train_only.add(rel)
    
    assert train_only.intersection(val_only).intersection(test_only) == set()
    return train_only, val_only, test_only


def tail_ontologies(relations: dict, min_freq: int, max_freq: int, limit: int, split) -> set:
    assert min_freq >= 0 and max_freq > 0 and limit > 0 and split in ["train", "val", "test"], "tail_ontologies(): error in parameters"
    result = set()
    relations_list = list(sorted(list(relations.items()), key=lambda x: x[1][split]))
    for rel, item in relations_list:
        if len(result) >= limit:
            break
        if min_freq < item[split] <= max_freq:
            result.add(rel)

    return result



def explore_kb(relations_input_file: str) -> Dict[str, List[str]]:
    relations: dict = read_json(relations_input_file)
    train_only, val_only, test_only = present_only(relations)

    tail_ontology_train = tail_ontologies(relations, min_freq=2, max_freq=float("inf"), limit=100, split="train")
    tail_ontology_dev = tail_ontologies(relations, min_freq=3, max_freq=float("inf"), limit=40, split="val")
    tail_ontology_test = tail_ontologies(relations, min_freq=3, max_freq=float("inf"), limit=40, split="test")

    tail_ontology_train = tail_ontology_train.difference(tail_ontology_dev).difference(tail_ontology_test)
    tail_ontology_dev = tail_ontology_dev.difference(tail_ontology_train).difference(tail_ontology_test)
    tail_ontology_test = tail_ontology_test.difference(tail_ontology_train).difference(tail_ontology_dev)

    train_ont, dev_ont, test_ont = list(tail_ontology_train), list(tail_ontology_dev), list(tail_ontology_test)

    train_ont.extend(list(train_only))
    dev_ont.extend(list(val_only))
    test_ont.extend(list(test_only))

    return {"train_out_of_ontology": sorted(train_ont), "dev_out_of_ontology": sorted(dev_ont), "test_out_of_ontology": sorted(test_ont)}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='seed to fix the randomness (default 42)', default=42)
    parser.add_argument('--kb_path', type=str, help='path to kb.json (default data/kb.json)', default="data/kb.json")
    parser.add_argument('--kqa_input_path', type=str, help='path to kb.json (default "data/data_split_60_3.json")', default="data_split_60_3")
    parser.add_argument('--relations_output_path', type=str, help='path to kb.json (default data/kb.json)', default="data/relations.json")
    args = parser.parse_args()

    build_relations(args.kb_path, args.kqa_input_path, args.relations_output_path)

    random.seed(args.seed)
    output: dict = explore_kb(args.relations_output_path)

    write_json("ontology_split.json", output)

    print("ontology_to_drop = ", output["train_out_of_ontology"])
    print()
    print("ontology_dev_specific = ", output["dev_out_of_ontology"])
    print()
    print("ontology_test_specific = ", output["test_out_of_ontology"])













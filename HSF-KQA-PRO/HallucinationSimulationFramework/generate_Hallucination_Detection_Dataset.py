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


def hallucinate_the_mrl(line: dict, keep_mrl: bool, autodetect: bool = False) -> dict:
    if keep_mrl:
        line["is_hallucinations"] = 1
    else:
        line["program"] = []
    
    if autodetect:
        line["program"] = [
                {"function": "<missing_ontology>", "dependencies": [], "inputs": []}
            ]  # we replace the program seq with an empty program seq, for the selected ids
        line["is_hallucinations"] = 1
    return line

def filter_program(program: List[dict], ontologies) -> bool:
    return any([ontology in function["inputs"] for function in program for ontology in ontologies])

def collect_out_of_ontology_question(dataset: List[dict], ontology_to_collect: List[str], ontology_to_avoid: List[str], keep_mrl: bool, autodetect: bool = False) -> List[dict]:
    result, indexes= [], []
    for idx, line in enumerate(dataset):
        program: List[str] = line["program"]
        
        if keep_mrl and "is_hallucinations" not in line:
            line["is_hallucinations"] = 0
        if program == []:
            indexes.append(idx)
            continue
        is_to_collect = filter_program(program, ontology_to_collect)
        is_to_avoid = filter_program(program, ontology_to_avoid)
        if is_to_avoid:
            indexes.append(idx)
            continue
        elif is_to_collect:
            line = hallucinate_the_mrl(line, keep_mrl=keep_mrl, autodetect=autodetect)
            result.append(line)
    return result 

def collect_in_ontology_question(dataset: List[dict], ontology_to_avoid: List[str]) -> List[dict]:
    # this function return only dataset samples that are in-ontology.
    result, indexes = [], []
    for idx, line in enumerate(dataset):
        program: List[dict] = line["program"]
        if program == []:
            indexes.append(idx)
            continue
        is_to_avoid = filter_program(program, ontology_to_avoid)
        if is_to_avoid:
            indexes.append(idx)
            continue
        result.append(line)
    return result 



def activation_model_dataset_generator(
    ontology_to_drop: List[str],
    ontology_to_drop_extra_test: List[str],
    ontology_to_drop_extra_dev: List[str],
    input_folder: str,
    dest_folder: str,

) -> None:
    out_of_domain = True
    use_expands = True
    os.makedirs(dest_folder, exist_ok=True)

    train_ooo_questions_expands, dev_ooo_questions_expands, test_ooo_questions_expands = None, None, None
    for split in ["train", "val", "test"]:
        print(f"#===== {split} =====#")
        dataset_jsonl: List[Dict] = read_jsonl(f"{input_folder}/{split}.json")[0]
        print("dataset len", len(dataset_jsonl))
        in_domain = None
        if split == "train":
            '''
            Here we collect only the ooo questions, we create the the train split at val step.
            '''
            train_ooo_questions_expands = collect_out_of_ontology_question(dataset_jsonl, ontology_to_drop, ontology_to_drop_extra_dev +  ontology_to_drop_extra_test, keep_mrl=True)
            print("TRAIN OOO", len(train_ooo_questions_expands))
            dev_ooo_questions_expands = collect_out_of_ontology_question(dataset_jsonl, ontology_to_drop_extra_dev, ontology_to_drop + ontology_to_drop_extra_test, keep_mrl=True)
            test_ooo_questions_expands = collect_out_of_ontology_question(dataset_jsonl, ontology_to_drop_extra_test, ontology_to_drop + ontology_to_drop_extra_dev, keep_mrl=True)
            continue

        elif split == "val":
            '''
            we split the val into a two split HDD train and HDD dev for the in-ontology utterances.
            Hence, here we create both split.
            '''
            assert train_ooo_questions_expands is not None and dev_ooo_questions_expands is not None
            if out_of_domain:

                dataset_dev: List[Dict] = dataset_jsonl.copy()
                in_domain: List[Dict] = collect_in_ontology_question(
                    dataset_dev, ontology_to_drop + ontology_to_drop_extra_test + ontology_to_drop_extra_dev
                )

                dev_ooo_questions: List[dict] = collect_out_of_ontology_question(dataset_dev, ontology_to_drop_extra_dev, ontology_to_drop + ontology_to_drop_extra_test, keep_mrl=True)

                dev_in_domain = in_domain[: len(dev_ooo_questions)]
                print("final dev ooo len", len(dev_ooo_questions), "final in domain len", len(dev_in_domain))

                dev_preprocessed = dev_in_domain

                dev_preprocessed.extend(dev_ooo_questions)


                train_in_domain = in_domain[len(dev_ooo_questions) :]
                assert len(dev_preprocessed) == 2 * len(dev_ooo_questions)
                if use_expands:
                    dev_preprocessed.extend(dev_ooo_questions_expands)
                    len_ooo_expand = len(dev_ooo_questions_expands)
                    dev_in_domain_expand, train_in_domain = train_in_domain[:len_ooo_expand], train_in_domain[len_ooo_expand:]
                    dev_preprocessed.extend(dev_in_domain_expand)

                    count_out_of_domain, count_in_domain = 0, 0
                    for line in dev_preprocessed:
                        if line["program"] == [] or line["is_hallucinations"] == 1:
                            count_out_of_domain +=1
                        else:
                            count_in_domain += 1
                    print("expand dev: in domain", count_in_domain, "count_out_of_domain", count_out_of_domain, "final len", count_in_domain + count_out_of_domain)
                write_jsonl(f"{dest_folder}/val.json", dev_preprocessed)

            dataset_jsonl = disrupt_knowledge_(
                dataset_jsonl,
                ontology_to_drop_extra_test + ontology_to_drop_extra_dev,
                keep_ooo_questions_without_mrl=False,
            )


            ooo_questions: List[dict] = collect_out_of_ontology_question(dataset_jsonl, ontology_to_drop, ontology_to_drop_extra_dev +  ontology_to_drop_extra_test, keep_mrl=True)
            ooo_questions.extend(
                train_ooo_questions_expands
            )  # we add the train ooo-questions into the dev set, we will use it for train our HDM -classifier
            print("final train ooo len", len(ooo_questions), "final in domain len", len(train_in_domain))
            ooo_questions.extend(train_in_domain)
            print("FINAL train len", len(ooo_questions))
            write_jsonl(f"{dest_folder}/train.json", ooo_questions)

            count_out_of_domain, count_in_domain = 0, 0
            for line in ooo_questions:
                if line["program"] == [] or line["is_hallucinations"] == 1:
                    count_out_of_domain +=1
                else:
                    count_in_domain += 1
            print("train: in domain", count_in_domain, "count_out_of_domain", count_out_of_domain)


        elif split == "test":
            in_domain_test: List[Dict] = collect_in_ontology_question(
                    dataset_jsonl, ontology_to_drop +  ontology_to_drop_extra_dev + ontology_to_drop_extra_test
                )
            new_dataset = collect_out_of_ontology_question(dataset_jsonl, ontology_to_drop_extra_test, ontology_to_drop + ontology_to_drop_extra_dev, keep_mrl=True)

            new_dataset.extend(in_domain_test)
            new_dataset.extend(test_ooo_questions_expands)

            print("FINAL test len", len(new_dataset))
            write_jsonl(f"{dest_folder}/test.json", new_dataset)
            
            count_out_of_domain, count_in_domain = 0, 0
            for line in new_dataset:
                if line["program"] == [] or line["is_hallucinations"] == 1:
                    count_out_of_domain +=1
                else:
                    count_in_domain += 1
            print("test: in domain", count_in_domain, "count_out_of_domain", count_out_of_domain, "final len: ", count_in_domain + count_out_of_domain)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='path to the input folder, it assume the presence of train.jsonl, val.jsonl (default data_split/)', default="data_split_60_3/")
    parser.add_argument('--output_folder', type=str, help='output path with the new splits (default out_folder/out_folder/hdd-dataset)', default="out_folder/hdd-dataset/")
    parser.add_argument('--output_folder', type=str, help='output path with the new splits (default out_folder/out_folder/hdd-dataset)', default="out_folder/hdd-dataset/")
    args = parser.parse_args()

    ontology_split = read_json("ontology_split.json")
    ontology_to_drop = ontology_split["train"]
    ontology_dev_specific = ontology_split["dev"]
    ontology_test_specific = ontology_split["test"]

    activation_model_dataset_generator(ontology_to_drop, ontology_to_drop_extra_dev=ontology_dev_specific, ontology_to_drop_extra_test=ontology_test_specific, input_folder=args.input_folder, dest_folder=args.output_folder)



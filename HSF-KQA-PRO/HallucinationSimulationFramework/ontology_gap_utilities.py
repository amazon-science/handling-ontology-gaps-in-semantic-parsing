#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

from typing import *

import json



def is_to_filter(line: dict, keyword_list: List[str]) -> bool:
    str_dict: str = json.dumps(line["program"])
    if str_dict == "[]":  # special case for activation_model_dataset_generator()
        return True  # questo serviva quando io nel train ho rimosso qualcosa e poi deve essere rimosso anche nel dev e non riaggiunto.
    return any(keyword in str_dict for keyword in keyword_list)


def filter_program(program: List[dict], ontologies) -> bool:
    return any([ontology in function["inputs"] for function in program for ontology in ontologies])


def filter_knowledge(dataset, keyword_list: List[str]) -> List[Union[Tuple[int, dict], Dict]]:
    # No in place modification
    return [
        [idx, line]
        for idx, line in enumerate(dataset)
        if filter_program(line["program"], keyword_list)
    ]


def disrupt_knowledge_(
    dataset: List[dict], keyword_list: List[str], keep_ooo_questions_without_mrl=False, replace_with_unk=False, delete_matched_sentences=False
) -> List[dict]:
    """
    keep_ooo_questions_without_mrl: bool = if it is True we keep the sentences but we replace the program (mrl) with an empty sequence, otherwise we drop the completely the example.
    """
    sentence_to_drop: List[Tuple[int, dict]] = filter_knowledge(dataset, keyword_list)

    print("n sentence to drop", len(sentence_to_drop))
    sentence_to_drop.sort(
        key=lambda x: x[0], reverse=True
    )  # id descending order is used to delete the idx, to prevent id shifting.
    for idx, line in sentence_to_drop:

        if line["program"] == []:
            del dataset[idx]
            continue
        if replace_with_unk:
            dataset[idx]["program"] = [
                {"function": "<missing_ontology>", "dependencies": [], "inputs": []}
            ]  # we replace the program seq with an empty program seq, for the selected ids
            dataset[idx]["is_hallucinations"] = 1
        elif keep_ooo_questions_without_mrl:
            dataset[idx]["program"] = []  # we replace the program seq with an empty program seq, for the selected ids
            dataset[idx]["is_hallucinations"] = 1
        elif delete_matched_sentences:
            del dataset[idx]
        else:
            del dataset[idx]

    return dataset


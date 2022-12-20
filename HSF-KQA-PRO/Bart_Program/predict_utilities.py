'''
This code starts from the code of KQA-Pro_Baseline" (https://github.com/shijx12/KQAPro_Baselines) at the commit "7cea2738fd095a2c17594d492923ee80a212ac0f (4th October 2022) then some modifications are applied.
Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

We create some function around the KQA-PRO Baseline code in order to be executed multiple times.
'''

from typing import *
from itertools import product
import numpy as np
from rich.progress import track


def get_combination_level_and_layers(layer_levels, layer_names) -> List[Tuple[int, str]]:
    # return pair like [(0, 'out_proj'), (0, 'k_proj'), ..., (5, 'fc2')]
    level_layer_pairs: List[Tuple[int, str]] = list(product(list(layer_levels), layer_names))
    return level_layer_pairs


def decode_tokenizer(tokenizer, output_ids):
    return [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
            output_id in output_ids]


def evaluate_prediction_kb(answers, given_answer, is_hallucination_list, logging):
    count_impossible_to_find_kb, correct, correct_possible, correct_impossible, count_possible, count_impossible, count = 0, 0, 0, 0, 0, 0, 0
    count_hallucinated = 0
    ERROR_IMPOSSIBLE_TO_FIND_IN_THE_KB: str = '<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>'
    for ans, query_answer_label, is_hallucination in zip(answers, given_answer, is_hallucination_list):
        if ans != query_answer_label:
            ...
        if ans == None or ans == ERROR_IMPOSSIBLE_TO_FIND_IN_THE_KB:
            ans = ERROR_IMPOSSIBLE_TO_FIND_IN_THE_KB
            count_impossible_to_find_kb += 1

        if ans == query_answer_label:
            correct += 1
        count += 1
        if not is_hallucination:
            if ans == query_answer_label:
                correct_possible += 1
            count_possible += 1
        else:
            if ans == query_answer_label:
                correct_impossible += 1
            if ans is not None and ans != ERROR_IMPOSSIBLE_TO_FIND_IN_THE_KB:
                count_hallucinated += 1
            count_impossible += 1

    if count_impossible > 0:
        acc_impossible = correct_impossible / count_impossible
        hallucination_percentage = (count_hallucinated / count_impossible) * 100
        logging.info('count hallucinated: {}'.format(count_hallucinated))
        logging.info('hallucination percentage: {}'.format(hallucination_percentage))
        logging.info('acc - Impossible: {}'.format(acc_impossible))
    else:
        logging.info('This dataset has no impossible questions')
    logging.info('count_impossible_to_find_kb: {}'.format(count_impossible_to_find_kb))
    if count > 0:
        acc = correct / count
        logging.info('acc - ALL: {}'.format(acc))
    if count_possible > 0:
        acc_possible = correct_possible / count_possible
        logging.info('acc - Possible: {}'.format(acc_possible))



def query_kb(executor, outputs):
    answers = []
    for output in track(outputs, description="query kb", total=len(outputs)):
        chunks = output.split('<func>')
        func_list = []
        inputs_list = []
        for chunk in chunks:
            chunk = chunk.strip()
            res = chunk.split('<arg>')
            res = [_.strip() for _ in res]
            if len(res) > 0:
                func = res[0]
                inputs = []
                if len(res) > 1:
                    for x in res[1:]:
                        inputs.append(x)
                else:
                    inputs = []
                func_list.append(func)
                inputs_list.append(inputs)
        ans = executor.forward(func_list, inputs_list, ignore_error=True)
        if ans == None:
            ans = '<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>'
        answers.append(ans)
    return answers


def filter_failed_artificial_hallucinations(result: dict):
    # we remove all the stimulated hallucinations for which the resulting MRL is still correct.
    list_idx_to_remove: List[int] = []
    use_activation: bool = len(result["global_feature_list"]) != 0


    assert len(result["outputs"]) == len(result["labels"])
    for idx, (out, lab, is_hallucination) in enumerate(zip(result["outputs"], result["labels"], result["hallucination_bool_list"])):
        if out == lab and is_hallucination:
            list_idx_to_remove.append(idx)


    if use_activation:
        excel = np.array(result["global_feature_list"])
        excel = np.delete(excel, list_idx_to_remove, axis=0)
        result["global_feature_list"] = excel


    for idx in reversed(list_idx_to_remove):
        for key, item in result.items():
            if key == "global_feature_list" or len(item) == 0:
                continue
            del result[key][idx]

    if use_activation:
        assert result["global_feature_list"].shape[0] == len(result["perplexity_list"]), (result["global_feature_list"].shape[0], len(result["perplexity_list"]))

    return result








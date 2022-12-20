#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
import argparse
from collections import defaultdict
from enum import Enum
import glob
from typing import *
from dataclasses import dataclass
from os.path import getctime


import pandas as pd

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



@dataclass
class NSPModelOutput:
    pred: str
    label: str
    is_impossible: int
    ppl: float
    out_mrl: str
    label_mrl: str


class Exp_names(Enum):
    perplexity_no_kb_filter: str = "perplexity-no-kb-filter"
    activations_no_kb_filter: str = "activations-no-kb-filter"
    mcd_no_kb_filter: str = "mcd-no-kb-filter"
    perplexity_activations_no_kb_filter: str = "perplexity+activations-no-kb-filter"
    activations_mcd_no_kb_filter: str = "activations+mcd-no-kb-filter"
    perplexity_mcd_no_kb_filter: str = "perplexity+mcd-no-kb-filter"
    perplexity_activations_mcd_no_kb_filter: str = "perplexity+activations+mcd-no-kb-filter"
    perplexity: str = "perplexity"
    activations: str = "activations"
    mcd: str = "mcd"
    perplexity_activations: str = "perplexity+activations"
    activations_mcd: str = "activations+mcd"
    perplexity_mcd: str = "perplexity+mcd"
    perplexity_activations_mcd: str = "perplexity+activations+mcd"


def read_txt(filename: str) -> List[str]:
    with open(filename, "r") as reader:
        return [line.strip() for line in reader.readlines()]


def preprocess_output_nsp_model(output_nsp_model: List[str]) -> List[NSPModelOutput]:
    col_splits = [line.split("|||") for line in output_nsp_model]
    type_cast = [
        NSPModelOutput(
            str(pred), str(label), int(is_impossible), float(ppl), out_mrl, label_mrl
        )
        for (pred, label, is_impossible, ppl, out_mrl, label_mrl) in col_splits
    ]
    return type_cast



def kb_baseline_scorer(output_nsp_model, only_executable_mrls: bool):

    predictions, labels, predictions_h, labels_h, predictions_err, labels_err = [], [], [], [], [], []

    for out_nsp in output_nsp_model:
        impossible_to_find_in_kb: bool = out_nsp.pred == "<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>"
        
        if impossible_to_find_in_kb and only_executable_mrls:
            # we keep only executable MRLs
            continue

        mrl_pred, mrl_label, is_impossible = out_nsp.out_mrl, out_nsp.label_mrl, out_nsp.is_impossible == 1


        # Ontology gaps case
        if is_impossible:
            if mrl_pred == mrl_label:
                assert False
            else:
                if impossible_to_find_in_kb:
                    labels_h.append("reject")
                    predictions_h.append("reject")
                else:
                    labels_h.append("reject")
                    predictions_h.append("accept")
        else:
            if mrl_pred == mrl_label:
                # all the Gold MRLs are always executable in KQA-PRO
                labels_h.append("accept")
                predictions_h.append("accept")
            # else: NSP error case, we skip it here.

        # NSP error case
        if not is_impossible:
            if mrl_label == mrl_pred:
                if impossible_to_find_in_kb:
                    predictions_err.append("reject")
                    labels_err.append("accept")
                else:                    
                    predictions_err.append("accept")
                    labels_err.append("accept")
            else:
                if impossible_to_find_in_kb:
                    predictions_err.append("reject")
                    labels_err.append("reject")
                else:
                    predictions_err.append("accept")
                    labels_err.append("reject")
        # else: Ontology gaps case, we skip it here.


        if impossible_to_find_in_kb and is_impossible:
            predictions.append("reject")
            labels.append("reject")
        elif impossible_to_find_in_kb and not is_impossible:
            predictions.append("reject")
            labels.append("accept")
        elif mrl_pred == mrl_label:
            predictions.append("accept")
            labels.append("accept")
        else:
            # NSP error case
            predictions.append("accept")
            labels.append("reject")


    out_dict = classification_report(labels, predictions, output_dict=True)
    out_dict_hallucinations = classification_report(labels_h, predictions_h, output_dict=True)
    out_dict_errors = classification_report(labels_err, predictions_err, output_dict=True)

    return out_dict, out_dict_hallucinations, out_dict_errors




def scorer_fixed_perplexity(output_nsp_model, only_executable_mrls: bool, threshold: float):
    assert 0 <= threshold <= 1

    predictions, labels, predictions_h, labels_h, predictions_err, labels_err = [], [], [], [], [], []

    for out_nsp in output_nsp_model:
        perplexity_condition: bool = out_nsp.ppl >= threshold

        impossible_to_find_in_kb: bool = out_nsp.pred in ["<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>"]
        
        if impossible_to_find_in_kb and only_executable_mrls:
            # we skip the non_executable MRLs
            continue


        mrl_pred, mrl_label, is_impossible = out_nsp.out_mrl, out_nsp.label_mrl, out_nsp.is_impossible == 1


        if is_impossible:
            # Ontology gaps
            if mrl_pred == mrl_label:
                assert False
            else:
                if impossible_to_find_in_kb:
                    labels_h.append("reject")
                    predictions_h.append("reject")
                else:
                    labels_h.append("reject")
                    if perplexity_condition:
                        predictions_h.append("accept")
                    else:                        
                        predictions_h.append("accept")
        else:
            if mrl_pred == mrl_label:
                if impossible_to_find_in_kb:
                    assert False
                else:
                    
                    labels_h.append("accept")
                    if perplexity_condition:
                        predictions_h.append("accept")
                    else:
                        predictions_h.append("reject")
            else:
                pass

        if not is_impossible:
            if mrl_label == mrl_pred:
                if impossible_to_find_in_kb:
                    labels_err.append("accept")
                    predictions_err.append("reject")
                    
                else:
                    if perplexity_condition:                    
                        predictions_err.append("accept")
                    else:
                        predictions_err.append("reject")
                    labels_err.append("accept")
                
            else:
                

                if impossible_to_find_in_kb:
                    predictions_err.append("reject")
                    labels_err.append("reject")
                else:
                    if perplexity_condition:
                        predictions_err.append("accept")
                    else:
                        predictions_err.append("reject")
                    labels_err.append("reject")


        if is_impossible and not perplexity_condition:
            predictions.append("reject")
            labels.append("reject")
        elif impossible_to_find_in_kb: # and not is_impossible:
            predictions.append("reject")
            labels.append("reject")
        elif mrl_pred == mrl_label and perplexity_condition:
            predictions.append("accept")
            labels.append("accept")
        else:
            predictions.append("accept")
            labels.append("reject")


    out_dict = classification_report(labels, predictions, output_dict=True)
    out_dict_hallucinations = classification_report(labels_h, predictions_h, output_dict=True)
    out_dict_errors = classification_report(labels_err, predictions_err, output_dict=True)

    return out_dict, out_dict_hallucinations, out_dict_errors



def end_to_end_evaluation(output_nsp_model: List[NSPModelOutput], hdm_predictions, skip_kb_filter: bool, is_top_dataset:bool):
    assert len(output_nsp_model) == len(hdm_predictions), (
        len(output_nsp_model),
        len(hdm_predictions),
    )

    end_to_end_correct, end_to_end_incorrect = 0, 0
    count_errors, count_in_ontology = 0, 0

    labels, predictions, labels_h, predictions_h, labels_err, predictions_err = [], [], [], [], [], []

    for out_nsp, hdm_pred in zip(output_nsp_model, hdm_predictions):

        is_ontology_gap: bool = out_nsp.is_impossible == 1
        is_to_reject: bool = int(hdm_pred) == 1
        kb_answer_pred, kb_answer_label = out_nsp.pred, out_nsp.label
        mrl_pred, mrl_label = out_nsp.out_mrl, out_nsp.label_mrl

        is_rejected_by_kb: bool = kb_answer_pred in ["<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>"]

        if not is_ontology_gap and not is_rejected_by_kb:
            if mrl_label != mrl_pred:
                count_errors += 1
            count_in_ontology += 1
        

        if is_rejected_by_kb and skip_kb_filter:
            continue


        #### Ontology gaps stage
        if is_ontology_gap:
            if mrl_pred == mrl_label:
                # all the ontology gaps prediction never corresponds with the gold MRL.
                assert False
            else:
                if is_to_reject or is_rejected_by_kb:
                    labels_h.append("reject")
                    predictions_h.append("reject")
                else:
                    labels_h.append("reject")
                    predictions_h.append("accept")
        else:
            if mrl_pred == mrl_label:
                if is_rejected_by_kb:
                    # case in which the pred and gold MRL are the same, but the KB reject them
                    # there is an assumption in KQA-PRO, each the gold MRL is always executable
                    # for that reason, this condition is never reached.
                    assert False
                elif is_to_reject:
                    labels_h.append("accept")
                    predictions_h.append("reject")
                else:
                    labels_h.append("accept")
                    predictions_h.append("accept")
            else:
                # case for NSP error, so we have to skip it here.
                ...


        #### NSP Error stage
        if not is_ontology_gap:
            if mrl_label == mrl_pred:
                if is_rejected_by_kb:
                    # case in which the pred and gold MRL are the same, but the KB reject them
                    # there is an assumption in KQA-PRO, each the gold MRL is always executable
                    # for that reason, this condition is never reached.
                    assert False
                elif is_to_reject:
                    predictions_err.append("reject")
                    labels_err.append("accept")
                else:                    
                    predictions_err.append("accept")
                    labels_err.append("accept")
                
            else:
                if is_to_reject or is_rejected_by_kb:
                    predictions_err.append("reject")
                    labels_err.append("reject")
                else:
                    predictions_err.append("accept")
                    labels_err.append("reject")
        else:
            # Ontology gaps stage, we have to skip it here.
            ...


        if mrl_pred == mrl_label:
            if is_rejected_by_kb:
                assert False
            elif is_to_reject:
                predictions.append("reject")
                labels.append("accept")
            else:
                predictions.append("accept")
                labels.append("accept")

        else:  # mrl_pred != mrl_label
            if is_to_reject or is_rejected_by_kb:
                predictions.append("reject")
                labels.append("reject")
            else:
                if is_top_dataset:
                    continue
                end_to_end_incorrect += 1
                predictions.append("accept")
                labels.append("reject")



    out_dict = classification_report(labels, predictions, output_dict=True)
    out_dict_hallucinations = classification_report(labels_h, predictions_h, output_dict=True)
    out_dict_errors = classification_report(labels_err, predictions_err, output_dict=True)

    return out_dict, out_dict_hallucinations, out_dict_errors



def build_csv_results(classification_report_dict: dict, exp_name: str, output_file: str = "results.csv") -> None:
    
    cols: List[str] = ["Exp name", "Macro F1 Score", "Accuracy", "Precision Good MRL", "Precision Bad MRL", "Recall Good MRL", "Recall Bad MRL", "F1 Score Good MRL", "F1 Score Bad MRL"]
    df = pd.DataFrame(columns=cols)

    output_dictionary: Dict[str, float] = {
        "Exp name": exp_name,
        "Macro F1 Score": classification_report_dict["macro avg"]["f1-score"],
        "Accuracy": classification_report_dict["accuracy"],
        "Precision Good MRL": classification_report_dict["accept"]["precision"],
        "Precision Bad MRL": classification_report_dict["reject"]["precision"],
        "Recall Good MRL": classification_report_dict["accept"]["recall"],
        "Recall Bad MRL": classification_report_dict["reject"]["recall"],
        "Macro Precision": classification_report_dict["macro avg"]["precision"],
        "Macro Recall": classification_report_dict["macro avg"]["recall"],
        "F1 Score Good MRL": classification_report_dict["accept"]["f1-score"], 
        "F1 Score Bad MRL": classification_report_dict["reject"]["f1-score"],
    }

    # we check if the dictionary keys are the same of the pandas columns
    assert len(cols) == len(output_dictionary.keys()) and set(cols).difference(set(output_dictionary.keys())) == set()

    df = df.append(output_dictionary, ignore_index=True)
    df.to_csv(f"src/end_to_end_evaluations/results/{output_file}", index=False)


def compute_mean_seed(results, is_hallucination: bool, is_executable: bool, is_error: bool, is_top_dataset:bool):


    cols: List[str] = ["Exp name", "Macro F1 Score", "Accuracy", "Macro Precision", "Precision Good MRL", "Precision Bad MRL", "Macro Recall",  "Recall Good MRL", "Recall Bad MRL", "F1 Score Good MRL", "F1 Score Bad MRL"]

    output: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for exp_name, entry in results.items():
        for seed, class_report in entry.items():
            output[exp_name]["Macro F1 Score"].append(class_report["macro avg"]["f1-score"])
            output[exp_name]["Accuracy"].append(class_report["accuracy"])
            output[exp_name]["Macro Precision"].append(class_report["macro avg"]["precision"])
            output[exp_name]["Precision Good MRL"].append(class_report["accept"]["precision"])
            output[exp_name]["Precision Bad MRL"].append(class_report["reject"]["precision"])
            output[exp_name]["Recall Good MRL"].append(class_report["accept"]["recall"])
            output[exp_name]["Recall Bad MRL"].append(class_report["reject"]["recall"])
            output[exp_name]["Macro Recall"].append(class_report["macro avg"]["recall"])
            output[exp_name]["F1 Score Good MRL"].append(class_report["accept"]["f1-score"])
            output[exp_name]["F1 Score Bad MRL"].append(class_report["reject"]["f1-score"])

    df = pd.DataFrame(columns=cols)

    # normalization
    for exp_name, entry in output.items():
        for metric, items in entry.items():
            output[exp_name][metric] = f"{round(np.array(items).mean().item(), 3)} ± {round(np.array(items).std().item(), 3)}"

        output[exp_name]["Exp name"] = exp_name
    
        df = df.append(output[exp_name], ignore_index=True)

    output_filename = f"src/end_to_end_evaluations/results/{'top_' if is_top_dataset else ''}mean_seed{'_hallucinations' if is_hallucination else ''}{'_executable' if is_executable else ''}{'_error' if is_error else ''}.csv"
    df.to_csv(output_filename, index=False)
        
       
def find_best_confidence_score_threshold_val(path_output_nsp_model):

    output_nsp_model: List[NSPModelOutput] = preprocess_output_nsp_model(read_txt(path_output_nsp_model))

    threshold_list: Set[float] = {round(out.ppl, 3) for out in output_nsp_model}


    results = []
    skip_kb_filter = True
    for threshold in threshold_list:
        out_dict_baseline_ppl, out_dict_hallucinations_baseline_ppl, predictions_err_baseline_ppl = scorer_fixed_perplexity(output_nsp_model, only_executable_mrls=skip_kb_filter, threshold=threshold)
        out_dict_hallucinations_baseline_ppl = out_dict_hallucinations_baseline_ppl["macro avg"]["f1-score"]
        predictions_err_baseline_ppl = predictions_err_baseline_ppl["macro avg"]["f1-score"]
        out_dict_baseline_ppl = out_dict_baseline_ppl["macro avg"]["f1-score"]
        
        results.append({"threshold": threshold, "out_dict_baseline_ppl": out_dict_baseline_ppl, "out_dict_hallucinations_baseline_ppl": out_dict_hallucinations_baseline_ppl, "predictions_err_baseline_ppl": predictions_err_baseline_ppl} )


    best_error_threshold, best_error_score = 0.0, 0.0

    best_out_of_ontology_threshold, best_out_of_ontology_score = 0.0, 0.0

    best_mixed_threshold, best_mixed_score = 0.0, 0.0


    for res in results:

        out_dict_baseline_ppl, out_dict_hallucinations_baseline_ppl, predictions_err_baseline_ppl, threshold  = res["out_dict_baseline_ppl"], res["out_dict_hallucinations_baseline_ppl"], res["predictions_err_baseline_ppl"], res["threshold"]
        
        if out_dict_hallucinations_baseline_ppl > best_out_of_ontology_score:
            best_out_of_ontology_score, best_out_of_ontology_threshold = out_dict_hallucinations_baseline_ppl, threshold
        

        if predictions_err_baseline_ppl > best_error_score:
            best_error_score, best_error_threshold = predictions_err_baseline_ppl, threshold

        if out_dict_baseline_ppl > best_mixed_score:
            best_mixed_score, best_mixed_threshold = out_dict_baseline_ppl, threshold

    print(best_mixed_score, best_error_score, best_out_of_ontology_score)
    return {"out_dict_baseline_ppl": best_mixed_threshold, "best_error_threshold": best_error_threshold, "best_out_of_ontology_threshold": best_out_of_ontology_threshold}


def scorer(seeds: List[str], recompute_best_confidence_score: bool, path_output_nsp_model: str, ood_test: bool, group_exp_name: str):
    USE_TOP_DATASET = ood_test

    keep_only_executable_MRLs = True
    prefix: str = "experiments/"

    # BEST Threshold found
    BEST_CS_THRESHOLD: float = 0.981
    if recompute_best_confidence_score:
        BEST_CS_THRESHOLD = find_best_confidence_score_threshold_val(path_output_nsp_model.replace("val", "test").replace("dev", "test"))
        # print(BEST_CS_THRESHOLD) # 0.981

    output_nsp_model = read_txt(path_output_nsp_model)
    output_nsp_model: List[NSPModelOutput] = preprocess_output_nsp_model(output_nsp_model)


    collect_exp_results, collect_exp_results_hallucinations, collect_exp_results_errors = defaultdict(
        lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(dict))


    out_dict_baseline, out_dict_hallucinations_baseline, predictions_err_baseline = kb_baseline_scorer(
        output_nsp_model, only_executable_mrls=keep_only_executable_MRLs)


    threshold_ppl_res = []
    for threshold in [BEST_CS_THRESHOLD]:
        threshold_ppl_res.append((threshold, scorer_fixed_perplexity(output_nsp_model,
                                                                     only_executable_mrls=keep_only_executable_MRLs,
                                                                     threshold=threshold)))

    for entry in Exp_names:
        exp_name = entry.value
        print(exp_name)

        for seed in seeds:
            collect_exp_results["baseline"][seed] = out_dict_baseline
            collect_exp_results_hallucinations["baseline"][seed] = out_dict_hallucinations_baseline
            collect_exp_results_errors["baseline"][seed] = predictions_err_baseline

            for res in threshold_ppl_res:
                threshold, (
                out_dict_baseline_ppl, out_dict_hallucinations_baseline_ppl, predictions_err_baseline_ppl) = res
                collect_exp_results[f"fixed_CS {threshold}"][seed] = out_dict_baseline_ppl
                collect_exp_results_hallucinations[f"fixed_CS {threshold}"][
                    seed] = out_dict_hallucinations_baseline_ppl
                collect_exp_results_errors[f"fixed_CS {threshold}"][seed] = predictions_err_baseline_ppl

            path = f'{prefix}/{group_exp_name}/{exp_name}/seed-{seed}/*/*/checkpoints/predictions_test{"_top" if USE_TOP_DATASET else ""}.txt'

            path_activations_model_predictions = sorted(glob.glob(path, recursive=True), key=getctime, reverse=True)[0]
            # [0] we take the most recent checkpoint.

            hdm_predictions = read_txt(path_activations_model_predictions)

            out_dict, out_dict_hallucinations, predictions_err = end_to_end_evaluation(output_nsp_model,
                                                                                       hdm_predictions,
                                                                                       skip_kb_filter=keep_only_executable_MRLs,
                                                                                       is_top_dataset=USE_TOP_DATASET)

            collect_exp_results[exp_name][seed] = out_dict
            collect_exp_results_hallucinations[exp_name][seed] = out_dict_hallucinations
            collect_exp_results_errors[exp_name][seed] = predictions_err

        compute_mean_seed(collect_exp_results, is_hallucination=False, is_executable=keep_only_executable_MRLs, is_error=False,
                          is_top_dataset=USE_TOP_DATASET)

        if not USE_TOP_DATASET:
            compute_mean_seed(collect_exp_results_hallucinations, is_hallucination=True, is_executable=keep_only_executable_MRLs,
                              is_error=False, is_top_dataset=USE_TOP_DATASET)
            compute_mean_seed(collect_exp_results_errors, is_hallucination=False, is_executable=keep_only_executable_MRLs,
                              is_error=True, is_top_dataset=USE_TOP_DATASET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed_list', type=str, help='add a sequence of ten or more seed like "30, 31, 32, 33, 34, 35, 37, 38, 36, 39"',
                        default="30, 31, 32, 33, 34, 35, 37, 38, 36, 39")

    parser.add_argument("--path_output_nsp_model", type=str, help="path to the output of the NSP model, file called danger_answer_test.txt or TOP_danger_answer_test.txt")
    parser.add_argument('--recompute_best_confidence_score', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--ood_test', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--group_exp_name', type=str, help="add the group exp name used to train the HDM, as shown in the README")

    args = parser.parse_args()
    seeds_list: List[str] = args.seed_list.split(", ")
    scorer(seeds_list, args.recompute_best_confidence_score, args.path_output_nsp_model, args.ood_test, args.group_exp_name)

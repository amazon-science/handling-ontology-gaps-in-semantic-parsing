#!/bin/bash
#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: CC-BY-NC-4.0



python3 src/end_to_end_evaluations/end_to_end_eval_scripts.py --path_output_nsp_model "experiments/riprova55/danger_answer_test.txt" --group_exp_name riprova55
python3 src/end_to_end_evaluations/end_to_end_eval_scripts.py --ood_test --path_output_nsp_model "experiments/riprova55/TOP_danger_answer_test.txt" --group_exp_name riprova55
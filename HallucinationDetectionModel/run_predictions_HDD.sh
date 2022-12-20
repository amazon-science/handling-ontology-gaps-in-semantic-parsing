#!/bin/bash
#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: CC-BY-NC-4.0


PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-30/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-30/" &


PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-31/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-31/" &



PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-32/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-32/" &


PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-33/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-33/" &

PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-34/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-34/" &



PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-35/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-35/" &

PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-36/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-36/" &


PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-37/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-37/" &

PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-38/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-38/" &


PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/activations/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/activations+mcd-no-kb-filter/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/activations-no-kb-filter/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/perplexity/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[6] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+mcd-no-kb-filter/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[7] evaluation.model_checkpoint_path="experiments/riprova55/perplexity-no-kb-filter/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[0] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[1] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations-no-kb-filter/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[2] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd-no-kb-filter/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[3] evaluation.model_checkpoint_path="experiments/riprova55/mcd-no-kb-filter/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[4] evaluation.model_checkpoint_path="experiments/riprova55/mcd/seed-39/" &
PYTHONPATH=. python3 src/predict.py train.pl_trainer.gpus=[5] evaluation.model_checkpoint_path="experiments/riprova55/perplexity+activations+mcd/seed-39/" &





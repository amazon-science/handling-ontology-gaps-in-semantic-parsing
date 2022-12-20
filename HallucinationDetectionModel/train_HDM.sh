#!/bin/bash
#Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#SPDX-License-Identifier: CC-BY-NC-4.0

PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity+activations" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[0] train.seed=30,31,32,33,34,35,36,37,38,39, &

PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity+activations+mcd" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[1]  train.seed=30,31,32,33,34,35,36,37,38,39 data.add_mc_dropout=True &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity+activations+mcd-no-kb-filter" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[2]  train.seed=30,31,32,33,34,35,36,37,38,39 data.add_mc_dropout=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity+activations-no-kb-filter" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[3]  train.seed=30,31,32,33,34,35,36,37,38,39   data.no_kb_filter=True  &


PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[4]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_perplexity=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity+mcd" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[5]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_perplexity=True data.add_mc_dropout=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity-no-kb-filter" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[6]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_perplexity=True data.no_kb_filter=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/perplexity+mcd-no-kb-filter" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[7]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_perplexity=True data.no_kb_filter=True data.add_mc_dropout=True  &


PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/activations" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[0]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_activations=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/activations+mcd" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[1]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_activations=True data.add_mc_dropout=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/activations-no-kb-filter" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[2]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_activations=True data.no_kb_filter=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/activations+mcd-no-kb-filter" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[3]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_activations=True data.no_kb_filter=True data.add_mc_dropout=True  &


PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/mcd" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[4]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_mc_dropout=True  &
PYTHONPATH=. python3 src/train.py -m  train.model_name="riprova55/mcd-no-kb-filter" +n_jobs=10 hydra/launcher=joblib train.pl_trainer.gpus=[5]  train.seed=30,31,32,33,34,35,36,37,38,39   data.use_only_mc_dropout=True data.no_kb_filter=True  &


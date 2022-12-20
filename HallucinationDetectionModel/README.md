




# Hallucination Detection Model

This folder cointains the codebase to replicate the experiments performed using the Hallucination Detection Model.



## Setup
Install the required python-packages
```bash
pip3 install -r requirements.txt
```

## Train the HDM
After extracting the activations, perplexity or MCD values from the NSP mdoel it is possible to train the HDM.

You must first change the path of the train, dev, test set:
- This can be done using the Hydra command line
    ```bash
    PYTHONPATH=. python3 src/train.py --data.train_path="<REPLACE WITH YOURPATH>"
    ```
- Or creating/modifying the YAML file in the folder ```conf/data/```

To replicate our experiments:
```bash
bash train_HDM.sh
```

## Inference

For inference use the folling command:
```bash
PYTHONPATH=. python3 src/predict.py evaluation model_checkpoint_path="experiments/<GROUP_OF_EXP_NAME>/seed-<SEED-NUM>/<EXP_NAME>/" 
```

To replicate our experiments using the Hallucination Detection Dataset (ontology gaps + NSP Errors)
```bash
bash run_predictions_HDD.sh
```
and for out-of-domain detection
```bash
bash run_predictions_OOD.sh
```

## End-to-End evaluation

To run the evaluation use the following script
```bash
python3 src/end_to_end_evaluations/end_to_end_eval_scripts.py --path_output_nsp_model "<OUTPUT_NSP_MODEL>/danger_answer_test.txt" --group_exp_name "<GROUP_OF_EXP_NAME>"
```

To recompute the Fixed CS threshold use:
```bash
python3 src/end_to_end_evaluations/end_to_end_eval_scripts.py --recompute_best_confidence_score
```

Or using the bash file
```bash
bash scorer.sh
```

<br>
The output will be by the default in the folder ```src/end_to_end_evaluations/results/```



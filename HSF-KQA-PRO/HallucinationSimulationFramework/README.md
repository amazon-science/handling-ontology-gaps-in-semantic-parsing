
# Hallucination Simulation Framework (HSF)
This framework is used to build two dataset NSP in-ontology and Hallucination Detection Dataset and the OOD dataset.
This two dataset will be used with the NSP model to train it and then to extract the Hallucination Detection Features at inference-time.

## How to use it
1. Download the datasets
```bash
bash download_kqa-pro_dataset.sh
bash download_top_dataset.sh
```
2. You can rebuild from scratch the NSP In-ontology dataset and the Hallucination Detection Dataset otherwise you can use ones provided in the repository.
Edit the build_dataset.sh file with your path.
```bash
bash build_dataset.sh
```


## More details on the HSF pipeline



### Build new three split of the KQA-PRO dataset.
This is done because the KQA-PRO test set is provided without KoPL MRLs, so we merge the dev and training set a we create three new split.
With the percentage command you can specify an integer between 1 and 100, which will represent the percentage of the dev/test set.
```bash
python3 split_kqa-pro_dataset.py --input_folder data/KQAPro.IID/ --output_folder data_split_60 --percentage 20
```

### Generate NSP in-ontology dataset
The NSP in-ontology dataset, is used to train the NSP model. This dataset contains only in-ontology questions. 
```bash
python3 generate_NSP_in_ontology_dataset.py --input_folder data_split_60
```

### Generate Hallucination Detection Dataset (HDD)
To HDD is used with the NSP model to extract the Hallucination Detection Features.
```bash
python3 generate_Hallucination_Detection_Dataset.py --input_folder data_split_60
```

### Generate OOD dataset
The OOD dataset as for the HDD, is used to extract the Hallucination Detection Features.
You should download use the KQA-PRO dataset and the TOP v2 dataset.
```bash
python3 generate_OOD_dataset.py --input_folder_KQA data_split_60 --input_folder_TOPv2 "data/TOPv2_Dataset/"
```







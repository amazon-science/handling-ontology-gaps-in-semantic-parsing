#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
from typing import *
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class HDM_Dataset(Dataset):
    def __init__(
        self, file_path: str, balance_dataset: bool = False, use_only_perplexity: bool = False, use_only_activations: bool = False, no_kb_filter: bool = False, do_upsampling: bool = False, use_only_mc_dropout: bool = False, add_mc_dropout: bool = False):


        self.df = pd.read_csv(file_path)

        print(file_path.split("/")[-1])
        

        self._select_features(use_only_activations=use_only_activations, use_only_perplexity=use_only_perplexity, no_kb_filter=no_kb_filter, use_only_mc_dropout=use_only_mc_dropout, add_mc_dropout=add_mc_dropout)
        

        self._feature_list = list(self.df.columns.values)

        self.data = self.df.to_numpy()

        self._num_features = self.data.shape[1] - 1
        
        self.get_stats()
        if balance_dataset:
            # the ontology gap is the minority class, so we can downsampling the majority class
            self.data = self.data[self.data[:, -1].argsort(kind="mergesort")]
            len_data = self.data.shape[0]
            n_ones: int = np.count_nonzero(self.data[:, -1] )
            self.data = self.data[len_data-n_ones*2:, :]
            self.get_stats()

        elif do_upsampling:
            self._up_sampling()
            

        self._extract_features_and_labels()
            

    def _extract_features_and_labels(self):
        self.labels = self.data[:, -1]  # we extract the labels col
        self.data = self.data[:, :-1]


    def get_stats(self) -> None:
        print(f"Number of feature used {self._num_features}")
        print("POSITIVE CLASS:", np.count_nonzero(self.data[:, -1]))
        print("NEGATIVE CLASS:", np.count_nonzero(self.data[:, -1] == 0))

    def get_feature_list(self) -> List[str]:
        return self._feature_list

    



    def _select_features(self, use_only_perplexity: bool, use_only_activations: bool, no_kb_filter: bool, use_only_mc_dropout: bool, add_mc_dropout: bool) -> None:

        assert (use_only_perplexity and not use_only_activations) or (not use_only_perplexity and use_only_activations) or (not use_only_perplexity and not use_only_activations), "ActivationDataset: ERROR in _select_features()"
        


        col_list = list(self.df.columns.values)
        banned_patterns = ["median", "min", "max", "3", "mean", "mc_dropout", "negative_feedback"]
        if use_only_activations:
            banned_patterns.append("perplexity")
        if add_mc_dropout:
            banned_patterns.remove("mc_dropout")
        
        cols: List[str]  = [col for col in col_list if all([not bp in col for bp in banned_patterns])]
        
        if use_only_perplexity:
            cols = ["possible_to_find_kb", "perplexity", "mask"]
            #self.df=self.df[["mc_dropout", "possible_to_find_kb", "perplexity", "mask"]]
            if add_mc_dropout:
                cols.insert(0, "mc_dropout")
            self.df=self.df[cols]
        elif use_only_mc_dropout:
            self.df=self.df[["mc_dropout", "possible_to_find_kb", "mask"]]
        else:
            self.df = self.df[cols]
        
    
        self.df["mask"][self.df["possible_to_find_kb"] == True] = 1  # if the MRL cannot be execute in the KB -> it is an hallucinations 

        if no_kb_filter:
            self.df.drop("possible_to_find_kb", axis=1, inplace=True)


        print("USED FEATURES: ", list(self.df.columns.values))


    def _up_sampling(self) -> None:
        print("--- start upsampling ---")
        self.data = self.data[self.data[:, -1].argsort(kind="mergesort")]
        n_zeros: int = np.count_nonzero(self.data == 0)
        minority_class = self.data[self.data[:, -1] == 0]
        minority_class_copy = minority_class.copy()
        print("minority class", minority_class_copy.shape, n_zeros)
        self.data = self.data[: n_zeros * 3, :]
        n_zeros: int = np.count_nonzero(self.data == 0)
        self.data = np.vstack([self.data, minority_class_copy])
        assert len(self.data) == n_zeros * 4, f"activation_dataset _up_sampling() error: in upsampling, {self.data.shape} instead of {n_zeros * 4}"


    @staticmethod
    def _load_dataset(file_path: str) -> np.array:
        with open(file_path, "r") as reader:  # we open the csv file in read-only mode
            reader.readline()  # skip the header
            data: np.array = np.loadtxt(fname=reader, delimiter=",")  # we load the csv into a numpy array
            return data

    def __len__(self) -> int:
        if self.data is None:
            raise Exception("You should call _load_dataset()")
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        if self.data is None:
            raise Exception("You should call _load_dataset()")
        return {"features": self.data[idx], "labels": self.labels[idx]}


if __name__ == "__main__":
    ad = HDM_Dataset("/home/ubuntu/projects/KQAPro_Baselines/activations.csv")
    print("Line 2: ", ad[2])
    print("Dataset len:", len(ad))

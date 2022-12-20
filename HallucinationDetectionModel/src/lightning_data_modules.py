#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
from typing import Any, Union, List, Optional

from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from src.HDM_Dataset import HDM_Dataset


class LightningDataModule(pl.LightningDataModule):
    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf

    def prepare_data(self, *args, **kwargs):
        
        config = self.conf.data

        self.train_dataset: HDM_Dataset = HDM_Dataset(
            config.train_path,
            balance_dataset=config.balance_dataset_train,
            use_only_perplexity=config.use_only_perplexity,
            use_only_activations=config.use_only_activations,
            do_upsampling=config.do_upsampling_train,
            no_kb_filter=config.no_kb_filter,
            use_only_mc_dropout=config.use_only_mc_dropout,
            add_mc_dropout=config.add_mc_dropout,

        )

        self.conf.data.n_features = self.train_dataset._num_features
        self.conf.data.feature_list = self.train_dataset.get_feature_list()
        
        self.dev_dataset: HDM_Dataset = HDM_Dataset(
            config.validation_path,
            balance_dataset=config.balance_dataset_dev,
            use_only_perplexity=config.use_only_perplexity,
            use_only_activations=config.use_only_activations,
            do_upsampling=config.do_upsampling_dev,
            no_kb_filter=config.no_kb_filter,
            use_only_mc_dropout=config.use_only_mc_dropout,
            add_mc_dropout=config.add_mc_dropout,
        )

        self.test_dataset: HDM_Dataset = HDM_Dataset(
            config.test_path,
            balance_dataset=False,
            use_only_perplexity=config.use_only_perplexity,
            use_only_activations=config.use_only_activations,
            no_kb_filter=config.no_kb_filter,
            use_only_mc_dropout=config.use_only_mc_dropout,
            add_mc_dropout=config.add_mc_dropout,
        )
        
    def setup(self, stage: Optional[str] = None):
        ...

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.conf.data.batch_size,
            num_workers=self.conf.data.num_workers,
            shuffle=True,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dev_dataset, batch_size=self.conf.data.batch_size_dev, num_workers=self.conf.data.num_workers, shuffle=False,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset, batch_size=self.conf.data.batch_size_test, num_workers=self.conf.data.num_workers, shuffle=False,
        )

    def get_top_dataset(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        config = self.conf.data
        top_dataset: HDM_Dataset = HDM_Dataset(
            config.top_test_path,
            balance_dataset=False,
            use_only_perplexity=config.use_only_perplexity,
            use_only_activations=config.use_only_activations,
            no_kb_filter=config.no_kb_filter,
            use_only_mc_dropout=config.use_only_mc_dropout,
            add_mc_dropout=config.add_mc_dropout,
        )
        return top_dataset

    def move_batch_to_device(self, batch, device: Union[str, torch.device]):
        return {key: item.to(device) for key, item in batch.items()}

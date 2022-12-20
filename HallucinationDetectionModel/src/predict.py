#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

from typing import *

import omegaconf
import hydra
from os.path import getctime 
import os
import glob

import pytorch_lightning as pl
from pytorch_lightning import Trainer


from src.lightning_data_modules import LightningDataModule
from src.lightning_module import PytorchLightningDataModule
from src.utils.utilities import set_determinism_the_old_way, dump_predictions




def predict(conf: omegaconf.DictConfig, path_exp: str, is_top_dataset: bool) -> None:

    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(True)

    pl_data_module: LightningDataModule = LightningDataModule(conf)
    pl_data_module.prepare_data()

    pl_module: PytorchLightningDataModule = PytorchLightningDataModule(conf)

    
    path = hydra.utils.to_absolute_path(f'{path_exp}/checkpoints/*.ckpt')
    
    model_checkpoint = sorted(glob.glob(path, recursive=True), key=getctime, reverse=True)[0]

    pl_module = pl_module.load_from_checkpoint(model_checkpoint)


    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)


    predict_path: str = '/'.join(model_checkpoint.split("/")[:-1])
    predictions, labels = pl_module.predict_fn(conf, pl_data_module, is_top_dataset=is_top_dataset)
    dump_predictions(predictions, labels, predict_path, "test", is_top_dataset=is_top_dataset)


def replace_overide_hydra(conf):
    exp_name: str = conf.evaluation.model_checkpoint_path
    is_top_dataset: bool = conf.data.is_top_dataset
    top_test_path: bool = conf.data.top_test_path
    device: List[int] = conf.train.pl_trainer.gpus
    path_exp = hydra.utils.to_absolute_path(f'{exp_name}/*/*/')
    config_path = sorted(glob.glob(hydra.utils.to_absolute_path(f'{path_exp}/.hydra/config.yaml'), recursive=True), key=getctime, reverse=True)[0] # we took the [0] because is the oldest one

    conf_checkpoint = omegaconf.OmegaConf.load(config_path)

    conf_checkpoint.data.top_test_path = top_test_path
    conf_checkpoint.data.is_top_dataset = is_top_dataset
    conf_checkpoint.train.pl_trainer.gpus = device
    conf_checkpoint.evaluation.model_checkpoint_path = exp_name


    return conf_checkpoint, path_exp


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):

    
    conf, path_exp = replace_overide_hydra(conf)
    

    if conf.data.is_top_dataset:
        conf.data.test_path = conf.data.top_test_path


    predict(conf, path_exp, conf.data.is_top_dataset)


if __name__ == "__main__":
    main()

#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

from typing import *
import json

import omegaconf
import hydra

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from src.lightning_data_modules import LightningDataModule
from src.lightning_module import PytorchLightningDataModule
from src.utils.utilities import set_determinism_the_old_way, dump_results


def train(conf: omegaconf.DictConfig) -> None:
    pl.seed_everything(conf.train.seed)
    set_determinism_the_old_way(True)

    pl_data_module: LightningDataModule = LightningDataModule(conf)
    pl_data_module.prepare_data()

    pl_module: PytorchLightningDataModule = PytorchLightningDataModule(conf)
    assert not conf.evaluation.model_checkpoint_path in ["", None], "Fix the conf/evaluation config"
    pl_module = pl_module.load_from_checkpoint(conf.evaluation.model_checkpoint_path)
    print(pl_module.model.__dict__)

    trainer: Trainer = hydra.utils.instantiate(conf.train.pl_trainer)
    
    test_output = trainer.test(pl_module, datamodule=pl_data_module)
    dump_results(conf.evaluation.model_checkpoint_path, test_output, "test")
    pl_module.init_metrics()

    val_output = trainer.test(pl_module, dataloaders=pl_data_module.val_dataloader())
    dump_results(conf.evaluation.model_checkpoint_path, val_output, "dev")
    pl_module.init_metrics()

    train_output = trainer.test(pl_module, dataloaders=pl_data_module.train_dataloader())
    dump_results(conf.evaluation.model_checkpoint_path, train_output, "train")
    pl_module.init_metrics()


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()

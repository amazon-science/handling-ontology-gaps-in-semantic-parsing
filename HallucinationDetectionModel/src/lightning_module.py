#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""

from typing import *
from src.HDM_Dataset import HDM_Dataset
from sklearn.metrics import f1_score

import torch
import hydra
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import omegaconf
from torchmetrics import F1Score, Precision, Recall, AUROC, AveragePrecision
from sklearn.metrics import f1_score
from src.hallucination_detection_model import HallucinationDetectionModel


class PytorchLightningDataModule(pl.LightningDataModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__()
        self.conf = conf
        self.save_hyperparameters(conf)
        self.model: HallucinationDetectionModel = HallucinationDetectionModel(
            self.conf.data.n_features, self.conf.model.first_layer_dim, self.conf.model.second_layer_dim
        )
        self.init_metrics()
        self.loss = torch.nn.CrossEntropyLoss()

    def init_metrics(self) -> None:
        self.f1_score: Dict[str, F1Score] = torch.nn.ParameterDict({"val": F1Score(num_classes=2, average="macro"), "test": F1Score(num_classes=2, average="macro")})
        self.precision_score: Dict[str, Precision] = torch.nn.ParameterDict({"val": Precision(num_classes=2, average="none"), "test": Precision(num_classes=2, average="none")})
        self.recall_score: Dict[str, Recall] = torch.nn.ParameterDict({"val": Recall(num_classes=2, average="none"), "test": Recall(num_classes=2, average="none")})
        self.auroc_score: Dict[str, Recall] = torch.nn.ParameterDict({"val": AveragePrecision(num_classes=2, average=None), "test": AveragePrecision(num_classes=2, average=None)})
        self.auroc_score_macro: Dict[str, Recall] = torch.nn.ParameterDict({"val": AveragePrecision(num_classes=2, average="macro"), "test": AveragePrecision(num_classes=2, average="macro")})

    def forward(self, **kwargs) -> dict:
        return self.model.forward(kwargs["features"].float())

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        loss = self.loss(forward_output, batch["labels"].long())
        self.log("loss", loss)
        return loss

    def shared_step(self, stage: str, batch: dict) -> dict:
        assert stage in ["val", "test"], f"shared_step(): ERROR -> {stage} is not in our stage list ['val', 'test']"
        forward_output = self.forward(**batch)
        loss = self.loss(forward_output, batch["labels"].long())
        preds, labels = self.compute_metrics(stage, forward_output, batch["labels"])
        self.log_dict({f"{stage}_loss": loss}, prog_bar=True)
        return {f"{stage}_loss": loss, f"{stage}_preds": preds, f"{stage}_labels": labels}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        return self.shared_step("val", batch)

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        return self.shared_step("test", batch)

    def shared_epoch_end(self, stage: str, outputs: List[torch.Tensor]) -> dict:

        assert stage in ["val", "test"], f"shared_epoch_end(): ERROR -> {stage} is not in our stage list ['val', 'test']"
        batch_losses = [x[f"{stage}_loss"] for x in outputs]  # This part
        preds = [item for sublist in outputs for item in sublist[f"{stage}_preds"].tolist()]
        labels = [item for sublist in outputs for item in sublist[f"{stage}_labels"].tolist()]
        f1 = f1_score(labels, preds, average='macro')        
        epoch_loss = torch.stack(batch_losses).mean()
        metrics_dict = {**self.per_class_metrics(stage), **{f"{stage}_loss_avg": epoch_loss.item()}}
        self.log_dict(metrics_dict, prog_bar=True, on_epoch=True, on_step=False)
        return {f"{stage}_loss_avg": epoch_loss.item(), f"{stage}_f1_avg": f1}

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end("test", outputs)

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end("val", outputs)



    def compute_metrics(self, stage: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
        assert stage in ["val", "test"], f"compute_metrics(): ERROR -> {stage} is not in our stage list ['val', 'test']"
        labels = labels.int()
        preds = logits.softmax(dim=-1).argmax(dim=-1)
        f1_score = self.f1_score[stage](preds, labels)
        recall = self.recall_score[stage](preds, labels)
        precision = self.precision_score[stage](preds, labels)
        auroc = self.auroc_score[stage](logits, labels)
        auroc_macro = self.auroc_score_macro[stage](logits, labels)

        return preds, labels


    def per_class_metrics(self, stage: str) -> dict:
        assert stage in ["val", "test"]
        epoch_f1 = self.f1_score[stage].compute()
        epoch_precision = self.precision_score[stage].compute()
        epoch_recall = self.recall_score[stage].compute()
        epoch_auroc = self.auroc_score[stage].compute()
        epoch_auroc_macro = self.auroc_score_macro[stage].compute()
        
        # Reseting internal state such that metric ready for new data
        [metric[stage].reset() for metric in [self.f1_score, self.precision_score, self.recall_score,self.auroc_score, self.auroc_score_macro]]

        return {
            f"{stage}_f1_avg": epoch_f1,
            f"{stage}_recall_class0": epoch_recall[0],
            f"{stage}_recall_class1": epoch_recall[1],
            f"{stage}_precision_class0": epoch_precision[0],
            f"{stage}_precision_class1": epoch_precision[1],
            f"{stage}_auroc_class0": epoch_auroc[0],
            f"{stage}_auroc_class1": epoch_auroc[1],
            f"{stage}_auroc_macro": epoch_auroc_macro,
        }


    def configure_optimizers(self):
        return hydra.utils.instantiate(self.conf.model.optimizer, params=self.model.parameters())

    def predict_fn(self, conf: omegaconf.DictConfig, pl_data_module, is_top_dataset: bool):
        

        result, labels = [], []

        device = conf.train.pl_trainer.gpus[0]

        self = self.to(device)

        dataloader = pl_data_module.test_dataloader()

        if is_top_dataset:
            dataloader = DataLoader(
                HDM_Dataset(
                            conf.data.top_test_path,
                            balance_dataset=conf.data.balance_dataset_train,
                            use_only_perplexity=conf.data.use_only_perplexity,
                            use_only_activations=conf.data.use_only_activations,
                            do_upsampling=conf.data.do_upsampling_train,
                            no_kb_filter=conf.data.no_kb_filter,
                            use_only_mc_dropout=conf.data.use_only_mc_dropout,
                            add_mc_dropout=conf.data.add_mc_dropout,
                )
                , shuffle=False, batch_size=2048)


        for batch in tqdm(dataloader, desc="predict"):
            batch = pl_data_module.move_batch_to_device(batch, device)
            logits = self.model.forward(batch["features"].float())
            preds = logits.softmax(dim=-1).argmax(dim=-1).tolist()
            
            result.extend(preds)
            labels.extend(batch["labels"])
        return result, labels

#!/usr/bin/python3
"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: CC-BY-NC-4.0
"""
from collections import defaultdict
from typing import *


import torch
import torch.nn as nn


from transformers import BartTokenizer, BartModel, BartForConditionalGeneration



class BARTFeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layer_names: List[str], layer_levels: List[int], use_encoder: bool = True):
        super().__init__()

        self.model = model if isinstance(model, BartModel) else model.model  # model.model: Enable compatability with BartForConditionalGeneration etc
        layer_levels.sort()
        self.bart_encoder: list = self.model.encoder.layers if use_encoder else self.model.decoder.layers
        n_layers_level_bart: int = len(self.bart_encoder)
        assert 0 <= layer_levels[-1] <= n_layers_level_bart and 0 <= layer_levels[0] < n_layers_level_bart, f"Levels in the range of [{layer_levels[0]} {layer_levels[-1]}] cannot be used, BART has only {n_layers_level_bart}"
        self.layers_name = layer_names # Layer name to track 
        self.layer_levels = layer_levels # Levels to track 
        self._features: Dict[int, Dict[str, torch.Tensor]] = defaultdict(lambda: defaultdict(lambda: torch.empty(0))) 
        
        for level in layer_levels:
            bart_level_layers = self.bart_encoder[level].__dict__["_modules"]
            BARTFeatureExtractor.add_attention_layers(bart_level_layers)
            for layer_name, layer in bart_level_layers.items():
                if layer_name in layer_names:
                    layer.register_forward_hook(self.save_outputs_hook(level, layer_name))
    

    @staticmethod
    def add_attention_layers(bart_level_layers: dict):
        attention_layer_name_list: List[str] = ["k_proj", "v_proj", "q_proj", "out_proj"]
        bart_attention_layers = bart_level_layers["self_attn"].__dict__["_modules"]
        bart_level_layers.update({layer_name: bart_attention_layers[layer_name] for layer_name in attention_layer_name_list})
        
    def get_features(self) -> dict:
        return self._features

    def get_features_level_and_layer_specific(self, level: int, layer_name: str, feature_sentence_level: bool = True) -> torch.Tensor:
        # feature_sentence_level = False, you will get features word pieces level, otherwise we compute the mean to obtain sentence level feature
        assert level in self.layer_levels, f"The {level} level is not recorded"
        assert layer_name in self.layers_name, f"This layer name {layer_name} is not recorded"
        
        return self._features[level][layer_name].mean(dim=1) if feature_sentence_level else self._features[level][layer_name]


    def save_outputs_hook(self, level: int, layer_name: str) -> Callable:
        def hook_func(_, __, output: torch.Tensor):
            self._features[level][layer_name] = output#.cpu()
            #self._features[level][layer_name] = self._features[level][layer_name] # The internal weights are not effected by this. now the batch shape is (n_batch, seq_len, hidden_dim)
        return hook_func


    def init_vectors(self):
        self._features = defaultdict(lambda: defaultdict(lambda: torch.empty(0))) # mapping layer_level -> layer_name -> features
        return self._features

if __name__ == "__main__":
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model_name)

    bfe = BARTFeatureExtractor(model, layer_names=["self_attn_layer_norm"], layer_levels=[3, 4])
    example_english_phrase = "UN Chief Says There Is No <mask> in Syria"
    batch = tokenizer(example_english_phrase, return_tensors="pt")
    generated_ids = model.generate(batch["input_ids"])
    out_features = bfe.get_features_level_and_layer_specific(4, "self_attn_layer_norm", feature_sentence_level=False)
    print(out_features.shape)

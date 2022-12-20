"""
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
"""


"""
Full KB:
https://drive.google.com/file/d/1CFq8yce_TP1ZbzN38CPse4bOl7ofrnBY/view?usp=sharing
"""
import argparse
from os import path
import json

import torch

from transformers import BartForConditionalGeneration, BartTokenizer

from Bart_Program.executor_rule import RuleExecutor
from Bart_Program.predict_utilities import query_kb


BASE_DIR = path.dirname(path.realpath(__file__))
KB_PATH = path.join(BASE_DIR, 'dataset/kb_full.json')


def cli(model_dir, device='cpu'):
    tokenizer = BartTokenizer.from_pretrained(model_dir)
    
    model = BartForConditionalGeneration.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()
    
    vocab = json.load(open(path.join(model_dir, 'vocab.json')))
    executor = RuleExecutor(vocab, KB_PATH)
    
    while True:
        text = input('\nUTTERANCE: ')
        with torch.no_grad():
            input_ids = tokenizer.batch_encode_plus([text], max_length=512, pad_to_max_length=True, return_tensors="pt", truncation=True)
            source_ids = input_ids['input_ids'].to(device)
            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
                num_beams=4
            )
            outputs = [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output_id in outputs]
            print(f"UNDERSTAND: {outputs}")
            
            answer = query_kb(executor, outputs)
            print(f"ANSWER: {answer}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args()
    
    cli(args.checkpoint_dir)

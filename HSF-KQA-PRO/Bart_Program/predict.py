'''
This code starts from the code of KQA-Pro_Baseline" (https://github.com/shijx12/KQAPro_Baselines) at the commit "7cea2738fd095a2c17594d492923ee80a212ac0f (4th October 2022) then some modifications are applied.
Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Our modifications consist in adding the code to extracts the Hallucination Detection Features.
'''
import os
from itertools import product
### andbac
from typing import *
### end andbac

import torch
import argparse
from tqdm import tqdm

from utils.misc import seed_everything
from utils.load_kb import DataForSPARQL
from .data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import logging
import time

import numpy as np
import pandas as pd

from Bart_Program.executor_rule import RuleExecutor

from Bart_Program.hook_class_wrapper import BARTFeatureExtractor
from Bart_Program.predict_utilities import query_kb, evaluate_prediction_kb, decode_tokenizer, \
    get_combination_level_and_layers, filter_failed_artificial_hallucinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings

warnings.simplefilter("ignore")  # hide warnings that caused by invalid sparql query


def compute_activations(bfe: BARTFeatureExtractor, layer_levels: list, layer_names: list) -> np.array:
    '''
    This function retrieve the NSP model activations and compute summary statistics (e.g., var, mean, median, ...)
    it returns a numpy array with that features
    the features follow this order: mean, var, median, min, max
    the output tensor is # batch_size,
    '''
    features_list = []

    for (level, layer_name) in list(product(layer_levels, layer_names)):
        features = bfe.get_features_level_and_layer_specific(level, layer_name,
                                                             feature_sentence_level=True)  # shape = batch_size, hidden_dim

        features_var = features.var(dim=1).unsqueeze(1)
        features_mean = features.mean(dim=1).unsqueeze(1)  # batch size x 1

        features_median = features.median(dim=1).values.unsqueeze(1)
        features_min = features.min(dim=1).values.unsqueeze(1)
        features_max = features.max(dim=1).values.unsqueeze(1)

        features = torch.cat((features_mean, features_var, features_median, features_min, features_max), dim=1)

        features_list.append(features.tolist())
    return torch.Tensor(features_list).permute((1, 0, 2)).flatten(1, 2).cpu().detach().numpy()


def compute_confidence_score(logits, normalize: bool = True, is_sequences_scores: bool = False) -> List[float]:
    # We consider the inverse of the posterior probability as perplexity
    # We normalize the probability by the length of the generated sequence
    # read more here about the normalization: https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
    # logits.sequences_scores shape == (batch_size, 1)
    sequences_scores = logits.sequences_scores if not is_sequences_scores else logits
    posterior_probability: torch.Tensor = torch.exp(sequences_scores)
    return posterior_probability.pow(1 / len(logits.scores)) if not normalize else posterior_probability


@torch.no_grad()
def compute_monte_carlo_dropout(model, source_ids, trials: int):
    model.train()
    perplexity_list, result = [], []

    for trial in range(trials):
        print(trial)
        outputs = model.generate(
            input_ids=source_ids,
            max_length=500,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=2
        )
        outputs.sequences = outputs.sequences.cpu()
        outputs = outputs.sequences_scores
        perplexity_normalized = compute_confidence_score(outputs, normalize=True, is_sequences_scores=True)
        perplexity_list.append(perplexity_normalized.tolist())
        outputs = None
        del outputs

    # refactor this loop
    for i in range(source_ids.shape[0]):
        result.append(np.var([perplexity_list[t][i] for t in range(trials)]).item())

    model.eval()
    return result


def compute_exact_match(predictions, labels, mask_list, kb_feedback: List[str], logging) -> Dict[str, float]:
    exact_match_total, exact_match_correct, hallucination_count = 0, 0, 0
    exact_match_total_possible_answer, exact_match_correct_possible_answer = 0, 0
    exact_match_executable_total, exact_match_executable_correct = 0, 0
    are_equal = []  # bool array 1 if the two mrl matche 0 otherwise.
    for label_program, pred_program, mask, kbf in zip(labels, predictions, mask_list, kb_feedback):
        if not mask:
            exact_match_correct_possible_answer += 1 if label_program == pred_program else 0
            exact_match_total_possible_answer += 1

            if kbf != "<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>":
                exact_match_executable_correct += 1 if label_program == pred_program else 0
                exact_match_executable_total += 1

        is_match: int = 1 if label_program == pred_program else 0
        exact_match_correct += is_match
        exact_match_total += 1
        are_equal.append(is_match)
    result = {"all_em": exact_match_correct / exact_match_total,
              "possible_em": exact_match_correct_possible_answer / exact_match_total_possible_answer,
              "is_correct_mrl_pred": are_equal}

    logging.info("Executable Exact Match {}".format(exact_match_executable_correct / exact_match_executable_total))
    logging.info("possible_correct_em {}".format(exact_match_correct_possible_answer))
    logging.info(
        "possible_not_correct_em {}".format(exact_match_total_possible_answer - exact_match_correct_possible_answer))
    logging.info("possible_total_em {}".format(exact_match_total_possible_answer))
    logging.info('Exact Match - Possible: {}'.format(result["possible_em"]))

    logging.info('Exact Match - ALL: {}'.format(result["all_em"]))
    return result


@torch.no_grad()
def predict(args, kb, model, data, device, tokenizer, executor, split: str):
    assert split in ["val", "train", "test"], f'{split} not in list ["val", "train", "test"]'
    model.eval()

    USE_ACTIVATIONS: bool = args.use_activations
    USE_MCD: bool = args.use_mcd
    USE_TOP_DATASET: bool = args.use_top_dataset

    top_prefix: str = "TOP_" if USE_TOP_DATASET else ""
    layer_levels, layer_names = list(range(0, 6)), ["out_proj", "k_proj", "v_proj", "q_proj", "fc1", "fc2"]
    level_layer_pairs = get_combination_level_and_layers(layer_levels, layer_names)  # List[Tuple[int, str]]
    all_outputs, all_answers, all_labels, all_questions = [], [], [], []
    hallucination_bool_list = []
    perplexity_list, global_feature_list, mc_dropout_list = [], [], []
    for idx_batch, batch in enumerate(tqdm(data, total=len(data), desc="predict")):
        source_mask, choices, target_ids, answer, is_hallucination = [x.cpu() for x in batch[1:]]

        source_ids = batch[0].to(device)

        is_hallucination = is_hallucination.squeeze(dim=1)
        hallucination_bool_list.extend(is_hallucination)

        if USE_ACTIVATIONS:
            bfe = BARTFeatureExtractor(model, layer_levels=layer_levels, layer_names=layer_names, use_encoder=True)

        outputs = model.generate(
            input_ids=source_ids,
            max_length=500,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=4
        )

        perplexity_normalized = compute_confidence_score(outputs, normalize=True)
        perplexity_list.extend(perplexity_normalized.tolist())

        if USE_ACTIVATIONS:
            features_list = compute_activations(bfe, layer_levels, layer_names)
            global_feature_list.append(features_list)

        outputs = outputs.sequences

        if USE_MCD:
            mc_dropout_list.extend(compute_monte_carlo_dropout(model=model, source_ids=source_ids, trials=30))

        outputs = outputs.cpu().numpy()
        source_ids = source_ids.cpu().numpy()

        all_labels.extend(target_ids.numpy())
        all_outputs.extend(outputs)
        all_answers.extend(answer.numpy())
        all_questions.extend(source_ids)

    outputs = decode_tokenizer(tokenizer, all_outputs)
    labels = decode_tokenizer(tokenizer, all_labels)
    questions = decode_tokenizer(tokenizer, all_questions)

    global_feature_list = global_feature_list if not USE_ACTIVATIONS else np.concatenate(global_feature_list, axis=0)

    result = {"outputs": outputs, "labels": labels, "questions": questions, "all_answers": all_answers,
              "hallucination_bool_list": hallucination_bool_list, "perplexity_list": perplexity_list,
              "global_feature_list": global_feature_list, "mc_dropout_list": mc_dropout_list}

    result = filter_failed_artificial_hallucinations(result)
    evaluate_and_dump_features(data, executor, result, logging, args, split, level_layer_pairs, top_prefix)
    return result


def get_negative_feedback_ids(hallucination_bool_list, outputs, labels):
    negative_feedback_ids = [idx for idx, (is_hallucination, label, prediction) in
                             enumerate(zip(hallucination_bool_list, labels, outputs)) if
                             (not is_hallucination and prediction != label)]
    for ids in negative_feedback_ids:
        hallucination_bool_list[ids] = 1

    return negative_feedback_ids, hallucination_bool_list


def dump_predictions_for_end2end_evaluation(args, top_prefix, split, result, predict_answer_kb, given_answer,
                                            mask_list):
    perplexity_list, outputs, labels, hallucination_bool_list = result["perplexity_list"], result["outputs"], result[
        "labels"], result["hallucination_bool_list"]
    hallucination_analysis = [(ans, query_answer_label, is_hallucination) for ans, query_answer_label, is_hallucination
                              in zip(predict_answer_kb, given_answer, hallucination_bool_list)]

    assert len(hallucination_analysis) == len(perplexity_list) and len(perplexity_list) == len(outputs), (
        len(hallucination_analysis), len(perplexity_list), len(outputs))

    with open(os.path.join(args.save_dir, f"{top_prefix}danger_answer_{split}.txt"), "w") as da_file:
        for (pred, label, possible_answer), ppl, output_mrl, label_mrl in zip(hallucination_analysis, perplexity_list,
                                                                              outputs, labels):
            line: str = "|||".join((pred, label, str(possible_answer), str(ppl), output_mrl, label_mrl))
            da_file.write(line + '\n')


def evaluate_and_dump_features(data, executor, predict_output: Dict[str, Any], logging, args, split, level_layer_pairs,
                               top_prefix: str):
    predict_output["hallucination_bool_list"] = [1 if is_hallucination else 0 for is_hallucination in
                                                 predict_output["hallucination_bool_list"]]
    hallucination_bool_list, outputs, labels = predict_output["hallucination_bool_list"], predict_output["outputs"], \
                                               predict_output["labels"]

    given_answer = [data.vocab['answer_idx_to_token'][a] for a in predict_output["all_answers"]]
    predict_answer_kb = query_kb(executor, predict_output["outputs"])

    evaluate_prediction_kb(predict_answer_kb, given_answer, hallucination_bool_list, logging)

    compute_exact_match(predictions=outputs, labels=labels, mask_list=hallucination_bool_list,
                        kb_feedback=predict_answer_kb, logging=logging)

    dump_predictions_for_end2end_evaluation(args, top_prefix, split, predict_output, predict_answer_kb, given_answer,
                                            mask_list=hallucination_bool_list)

    is_possible_to_find_in_the_kb: List[int] = [1 if ans == '<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>' else 0 for ans in
                                                predict_answer_kb]

    negative_feedback_ids, hallucination_bool_list_updated = get_negative_feedback_ids(hallucination_bool_list, outputs,
                                                                                       labels)
    predict_output["hallucination_bool_list"] = hallucination_bool_list_updated

    use_activations: bool = len(predict_output["global_feature_list"]) > 0

    use_mcd: bool = len(predict_output["mc_dropout_list"]) > 0

    if use_activations:
        final_dataframe = pd.DataFrame(predict_output["global_feature_list"],
                                       columns=[f"{layer_level}_{layer_names}_{type_aggreg}" for
                                                layer_level, layer_names in level_layer_pairs for type_aggreg in
                                                ["mean", "var", "median", "min", "max"]])
    else:
        final_dataframe = pd.DataFrame()

    if use_mcd:
        final_dataframe["mc_dropout"] = predict_output["mc_dropout_list"]

    final_dataframe["possible_to_find_kb"] = is_possible_to_find_in_the_kb
    final_dataframe["perplexity"] = predict_output["perplexity_list"]
    final_dataframe["negative_feedback"] = [1 if idx in negative_feedback_ids else 0 for idx in
                                            range(len(predict_output["perplexity_list"]))]

    final_dataframe["mask"] = predict_output["hallucination_bool_list"]

    folder_path = "./activations_folder/"

    os.makedirs(folder_path, exist_ok=True)
    final_dataframe.to_csv(f'{folder_path}/{split}.csv', index=False)

    print("--- DONE ---")


@torch.no_grad()
def predict_autodetect(args, kb, model, data, device, tokenizer, executor, split: str):
    model.eval()
    all_outputs, all_labels, all_questions, mask_list = [], [], [], []

    USE_TOP: bool = True

    top_prefix = "" if not USE_TOP else "_TOP"

    for idx_batch, batch in enumerate(tqdm(data, total=len(data), desc="predict")):
        # source_ids, source_mask, choices, target_ids, answer, mask_true_are_impossible = [x.to(device) if x is not None else x for x in batch]
        source_ids, source_mask, choices, target_ids, answer, mask_true_are_impossible = batch

        mask_list.extend(mask_true_are_impossible.tolist())

        outputs = model.generate(
            input_ids=source_ids.to(device),
            max_length=500,
            return_dict_in_generate=True,
            output_scores=True,
            num_beams=4
        )

        outputs = outputs.sequences

        all_labels.extend(target_ids.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())
        all_questions.extend(source_ids.cpu().numpy())

    outputs = decode_tokenizer(tokenizer, all_outputs)
    labels = decode_tokenizer(tokenizer, all_labels)
    questions = decode_tokenizer(tokenizer, all_questions)

    assert len(outputs) == len(labels) and len(labels) == len(questions), "Wrong decoding"

    # Here we dump the end-to-end QA predictions (predict.txt) and the resulting MRL (KOPL program) (predict_program.txt)
    with open(os.path.join(args.save_dir, f'predict_{split}{top_prefix}.txt'), 'w') as f, open(
            os.path.join(args.save_dir, f'predict_program_{split}{top_prefix}.txt'), 'w') as program_file, open(
            os.path.join(args.save_dir, f'label_program_{split}{top_prefix}.txt'), 'w') as program_labels, open(
            os.path.join(args.save_dir, f'text_program_{split}{top_prefix}.txt'), 'w') as program_text:
        for output, label, text in tqdm(zip(outputs, labels, questions)):
            program_file.write(output + '\n')
            program_labels.write(label + '\n')
            program_text.write(text + '\n')
            chunks = output.split('<func>')
            func_list = []
            inputs_list = []
            for chunk in chunks:
                chunk = chunk.strip()
                res = chunk.split('<arg>')
                res = [_.strip() for _ in res]
                if len(res) > 0:
                    func = res[0]
                    inputs = []
                    if len(res) > 1:
                        for x in res[1:]:
                            inputs.append(x)
                    else:
                        inputs = []
                    func_list.append(func)
                    inputs_list.append(inputs)
            ans = executor.forward(func_list, inputs_list, ignore_error=True)
            if ans == None:
                ans = '<ERROR-IMPOSSIBLE_TO_FIND_IN_THE_KB>'
            f.write(ans + '\n')


def vis(model, device, tokenizer, executor):
    model.eval()
    while True:
        # text = 'Who is the father of Tony?'
        # text = 'Donald Trump married Tony, where is the place?'
        text = input('Input your question:')
        with torch.no_grad():
            input_ids = tokenizer.batch_encode_plus([text], max_length=512, pad_to_max_length=True, return_tensors="pt",
                                                    truncation=True)
            source_ids = input_ids['input_ids'].to(device)

            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
                num_beams=4
            )
            outputs = [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                       output_id in outputs]
            print(outputs)
            print(query_kb(executor, outputs))

            # outputs = [post_process(output) for output in outputs]
            # print(outputs[0])


def train(args):
    args.test_file = args.test_file if args.test_file != "dev" else "val"
    device = args.device if torch.cuda.is_available() else 'cpu'
    USE_AUTODETECT: bool = args.autodetect

    logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, args.test_file)
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    #####val_loader = DataLoader(vocab_json, val_pt, args.batch_size)
    val_loader = DataLoader(vocab_json, val_pt, args.batch_size, training=False)
    vocab = train_loader.vocab
    kb = DataForSPARQL(os.path.join(args.input_dir, 'kb.json'))
    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(os.path.join(args.ckpt, 'tokenizer'))
    model = model_class.from_pretrained(os.path.join(args.ckpt, 'model'))
    model = model.to(device)
    # logging.info(model)
    rule_executor = RuleExecutor(vocab, os.path.join(args.input_dir, 'kb.json'))

    if args.cli:
        vis(model, device, tokenizer, rule_executor)
        exit(0)

    # validate(args, kb, model, val_loader, device, tokenizer, rule_executor)
    if not USE_AUTODETECT:
        predict(args, kb, model, val_loader, device, tokenizer, rule_executor, args.test_file.replace(".pt", ""))
    else:
        predict_autodetect(args, kb, model, val_loader, device, tokenizer, rule_executor,
                           args.test_file.replace(".pt", ""))
    # vis(args, kb, model, val_loader, device, tokenizer)


def validate(args, kb, model, data, device, tokenizer, executor):
    model.eval()
    count, correct = 0, 0
    exact_match_total, exact_match_correct = 0, 0
    with torch.no_grad():
        all_outputs = []
        all_answers = []
        for batch in tqdm(data, total=len(data)):
            source_ids, source_mask, choices, target_ids, answer, mask_true_are_impossible = [x.to(device) for x in
                                                                                              batch]
            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
            )

            all_outputs.extend(outputs.cpu().numpy())
            all_answers.extend(answer.cpu().numpy())
            exact_match_total += len(target_ids)
            for label, pred in zip(target_ids, outputs):
                label_program = tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                pred_program = tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                exact_match_correct += 1 if label_program == pred_program else 0

        outputs = [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                   output_id in all_outputs]
        assert exact_match_total > 0, f"validate() Error: exact_match_total is less than 0 {exact_match_total}"

        logging.info('KOPL Exact match: {}'.format(exact_match_correct / exact_match_total))

        given_answer = [data.vocab['answer_idx_to_token'][a] for a in all_answers]
        for a, output in tqdm(zip(given_answer, outputs)):
            chunks = output.split('<func>')
            func_list = []
            inputs_list = []
            for chunk in chunks:
                chunk = chunk.strip()
                res = chunk.split('<arg>')
                res = [_.strip() for _ in res]
                if len(res) > 0:
                    func = res[0]
                    inputs = []
                    if len(res) > 1:
                        for x in res[1:]:
                            inputs.append(x)
                    else:
                        inputs = []
                    func_list.append(func)
                    inputs_list.append(inputs)
            ans = executor.forward(func_list, inputs_list, ignore_error=True)
            if ans != a:
                ...
                # print(colored(output, 'red'))
                # print(func_list)
                # print(inputs_list)
            if ans == None:
                ans = 'no'
            if ans == a:
                correct += 1
            count += 1
        acc = correct / count
        logging.info('acc: {}'.format(acc))

        return acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--autodetect', default=False, action=argparse.BooleanOptionalAction)

    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--cli', required=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_activations', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--use_mcd', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--use_top_dataset', action=argparse.BooleanOptionalAction, default=False)

    # training parameters
    # parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--test_file', type=str, default="test.pt", help='test.pt or val.pt')
    #####parser.add_argument('--test_file', type=str, default="val.pt", help='test.pt or val.pt')
    parser.add_argument('--device', type=str, default="cuda", help='cuda:0 or cuda or cpu')

    # validating parameters
    # parser.add_argument('--num_return_sequences', default=1, type=int)
    # parser.add_argument('--top_p', default=)
    # model hyperparameters
    parser.add_argument('--dim_hidden', default=1024, type=int)
    parser.add_argument('--alpha', default=1e-4, type=float)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, '{}.predict.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    seed_everything(666)

    train(args)


if __name__ == '__main__':
    main()


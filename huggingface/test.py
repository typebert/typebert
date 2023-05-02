# coding=utf-8
# Copyright 2022 Kevin Jesse.
# Licensed under the CC-by license.

# Lint as: python3

import argparse
import logging
import os
import warnings
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    get_scheduler,
)

with open("labels.txt", 'r') as file:
    label_list = file.readlines()
    label_list = [line.rstrip() for line in label_list]

def flatten(t):
    return [item for sublist in t for item in sublist]

def get_labels(predictions, references, label_list, score_unk=False, top100=False):
    y_pred = predictions.detach().cpu().clone().numpy()
    y_true = references.detach().cpu().clone().numpy()

    if score_unk:
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        # if model predicts UNK, change UNK label to 'any' to score incorrectly.
        # true labels of 'any' do not exist in dataset so will not match an any prediction anyway.
        true_labels = [
            [label_list[l] if l != 0 else label_list[1] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]

    elif top100:
        # Only top100 types not unk or any
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100 and l > 1 and l < 102]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100 and l > 1 and l < 102]
            for pred, gold_label in zip(y_pred, y_true)
        ]
    else:
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]

    return true_predictions, true_labels


def compute_metrics(metric, metric_unk, metric_top100):
    results = metric.compute()
    results_unk = metric_unk.compute()
    results_top100 = metric_top100.compute()
    return {
        "precision_overall": results["overall_precision"],
        "recall_overall": results["overall_recall"],
        "f1_overall": results["overall_f1"],
        "accuracy_overall": results["overall_accuracy"],

        "precision_unk": results_unk["overall_precision"],
        "recall_unk": results_unk["overall_recall"],
        "f1_unk": results_unk["overall_f1"],
        "accuracy_unk": results_unk["overall_accuracy"],

        "precision_top100": results_top100["overall_precision"],
        "recall_top100": results_top100["overall_recall"],
        "f1_top100": results_top100["overall_f1"],
        "accuracy_top100": results_top100["overall_accuracy"],
    }


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default="typebert-model", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--model_name", default="kevinjesse/typebert", type=str,
                        help="The huggingface model architecture to be fine-tuned.")

    parser.add_argument("--tokenizer_name", default="kevinjesse/typebert", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    
    parser.add_argument("--dataset_name", default="kevinjesse/typebert", type=str,
                        help="Dataset for training/test. The typebert dataset is already configured for this script.")

    parser.add_argument("--do_eval", default=False, action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_valid", default=False, action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--checkpoint_path", default='kevinjesse/typebert', type=str,
                        help="Path to model checkpoint.")
    
    parser.add_argument("--drop_last", default=True, action='store_true',
                        help="Whether to drop the last uneven batch. For a final eval, this should be false.")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test(args)

def test(args):
    tokenized_hf = load_dataset(args.dataset_name)
    
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    tokenized_hf.set_format("pt", columns=['input_ids', 'labels'], output_all_columns=True)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint_path, num_labels=len(label_list))
    eval_dataset = tokenized_hf["test"]
    valid_dataset = tokenized_hf["validation"]
    logger = logging.getLogger(__name__)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, drop_last=args.drop_last)  #for exact numbers this should be changed.
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.eval_batch_size)

    device = accelerator.device
    print("Device: {0}".format(device))
    model.to(device)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader, valid_dataloader = accelerator.prepare(
        model, eval_dataloader, valid_dataloader
    )

    # Metrics - more detailed than overall accuracy in evaluator.py
    warnings.filterwarnings('ignore')
    metric = load_metric("seqeval")
    metric_unk = load_metric("seqeval")
    metric_top100 = load_metric("seqeval")

    eval_total_batch_size = args.eval_batch_size * accelerator.num_processes
    logger.info("***** Running test *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.eval_batch_size}")
    logger.info(f"  Total test batch size (w. parallel, distributed & accumulation) = {eval_total_batch_size}")

    completed_steps = 0
    progress_bar_eval = tqdm(range(len(eval_dataset) // eval_total_batch_size),
                     disable=not accelerator.is_local_main_process)
    export_predictions = []
    model.eval()

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(input_ids=batch['input_ids'], labels=None)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        preds, refs = get_labels(predictions_gathered, labels_gathered, label_list)
        export_predictions.extend(flatten(preds))
        preds_unk, refs_unk = get_labels(predictions_gathered, labels_gathered, label_list, score_unk=True)
        preds_100, refs_100 = get_labels(predictions_gathered, labels_gathered, label_list, top100=True)
        progress_bar_eval.update(1)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )
        metric_unk.add_batch(
            predictions=preds_unk,
            references=refs_unk,
        )
        metric_top100.add_batch(
            predictions=preds_100,
            references=refs_100,
        )


    eval_metric = compute_metrics(metric, metric_unk, metric_top100)
    accelerator.print(eval_metric)
    enums = list(map(str, list(range(len(export_predictions)))))
    export_predictions = list(map(str, export_predictions))
    export_predictions = ["{}\t{}".format(a_, b_) for a_, b_ in zip(enums, export_predictions)]
    with open(args.output_dir + "/test_predictions.txt", 'w') as f:
        f.write("\n".join(export_predictions))


if __name__ == "__main__":
    main()
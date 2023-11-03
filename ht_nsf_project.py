import sys
import os
import torch
from pathlib import Path
from transformers import TrainingArguments
from datasets import Dataset

from htriskpred.data import get_dataset, get_dataset_json, get_tokenizer_dataset
from htriskpred.model import get_model, train, test, compute_metrics, sample_explainer
import argparse

def set_dependent_config(args: TrainingArguments):
    if torch.cuda.is_available():
        args.fp16 = True
    args.gradient_checkpointing=model.supports_gradient_checkpointing

    return args


def main_train():
    mydataset = get_dataset_json(data_dir, 0.95, 0.05)
    tokenized_datasets = get_tokenizer_dataset(mydataset, tokenizer, task=task)
    
    default_args = {
        "output_dir": model_dir,
        "evaluation_strategy": "epoch",
        "num_train_epochs": 3,
        "logging_steps": 1,
        "log_level": "info",
        "report_to": "none",
    }

    training_args = set_dependent_config(TrainingArguments(
        per_device_train_batch_size=16,
        gradient_accumulation_steps=32,
        **default_args,
    ))

    train(
        training_args,
        model,
        tokenizer,
        tokenized_datasets,
        data_collator,
        compute_metrics,
    )
#     sample_explainer(
#         mydataset["train"][1551]["text"],
#         model,
#         tokenizer,
#         output_dir + "bert_vizNeg3.html",
#     )
    
    tokenized_datasets["train"].save_to_disk(train_data_dir)
    tokenized_datasets["validation"].save_to_disk(val_data_dir)
    tokenized_datasets["test"].save_to_disk(test_data_dir)


def main_test():
    eval_dataset = Dataset.load_from_disk(targeted_test)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_args = set_dependent_config(TrainingArguments(
        output_dir=model_dir,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=256,
        eval_accumulation_steps=256,
        dataloader_drop_last=False,
    ))

    metrics = test(model, eval_dataset, test_args, compute_metrics)

    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your Script Description")

#     parser.add_argument("folder_name", type=str, help="Name of the folder")
    parser.add_argument("--task", type=str, choices=["htrisk", "relatedness"], help="Task")
    parser.add_argument("--model-dir", type=str, help="model directory (output for TRAIN or input for TEST)")
    parser.add_argument("--type", type=str, choices=["train", "test"], help="TRAIN or TEST")
    parser.add_argument("--data-dir", type=str, help="Data directory for TRAIN")
    parser.add_argument("--checkpoint", type=str, help="Base model", default = "roberta-base")
    parser.add_argument("--data-type", type=str, choices=["train", "test", "val"], help="Data type for TEST")
    parser.add_argument("--base-dir", type=str, help="Base directory", default = "./")

    args = parser.parse_args()
    additional_tokens = [
        "[PHONE]",
        "[NAME]",
        "[LOCATION]",
        "[ONLYFANS]",
        "[SNAPCHAT]",
        "[USERNAME]",
        "[INSTAGRAM]",
        "[TWITTER]",
        "[EMAIL]"
    ]
    # TODO: Allow this as command line arguments
    task = args.task
    print(f"Task {task}")
    base_dir = args.base_dir
    model_dir = base_dir + args.model_dir
    train_data_dir = Path(model_dir) / "train_dataset"
    val_data_dir = Path(model_dir) / "val_dataset"
    test_data_dir = Path(model_dir) / "test_dataset"
    checkpoint = args.checkpoint
    if(args.type == "test"):
        checkpoint = model_dir
        targeted_test = Path(model_dir) / f"{args.data_type}_dataset"
    else:
        data_dir = base_dir + args.data_dir
    if not Path.exists(Path(model_dir)):
        Path.mkdir(Path(model_dir))
    model, tokenizer, data_collator = get_model(checkpoint, additional_tokens=additional_tokens)

    if args.type == "train":
        main_train()
    else:
        main_test()

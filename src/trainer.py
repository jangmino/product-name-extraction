from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import sys
import re
from datasets import Features, ClassLabel, Dataset
import os
import torch
import numpy as np

from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    Trainer,
)

from seqeval.metrics import f1_score

import logging


@dataclass
class ScriptArguments:
    cache_dir: Optional[str] = field(
        default="/Jupyter/huggingface/.cache", metadata={"help": "the cache dir"}
    )

    pretrained_model_name: Optional[str] = field(
        default="klue/roberta-large",
        metadata={"help": "the model name"},
    )

    dataset_path: Optional[str] = field(
        default="./local_data/supvervised_dataset.pkl",
        metadata={"help": "the dataset (pandas Dataframe pickle path)"},
    )


### Utilities ###
class_label = ClassLabel(num_classes=4, names=["O", "B-NAME", "I-NAME", "E-NAME"])
index2tag = {idx: tag for idx, tag in enumerate(class_label.names)}
tag2index = {tag: idx for idx, tag in enumerate(class_label.names)}


def find_sublist(main_list, sublist):
    """
    Find the starting index of a sublist within a main list.

    Args:
        main_list (list): The main list to search in.
        sublist (list): The sublist to find.

    Returns:
        int: The starting index of the sublist in the main list, or -1 if not found.
    """
    for i in range(len(main_list) - len(sublist) + 1):
        if main_list[i : i + len(sublist)] == sublist:
            return i
    return -1


def remove_leading_under_score_chr(in_str, under_score_chr=chr(9601)):
    """
    Removes the leading underscore character from each string in the input list if it matches the specified underscore character.

    Args:
        in_str (list): The list of strings to process.
        under_score_chr (str, optional): The underscore character to remove. Defaults to chr(9601).

    Returns:
        list: The list of strings with the leading underscore character removed if it matches the specified underscore character.
    """
    return [x[1:] if x[0] == under_score_chr else x for x in in_str]


def remove_leading_empty_like_str(in_str):
    """
    Removes leading empty-like characters from a string.

    Args:
        in_str (str): The input string.

    Returns:
        str: The input string with leading empty-like characters removed.
    """
    i = 0
    for x in in_str:
        if x != "":
            break
        i += 1
    return in_str[i:]


def label_token(
    tokens_source,
    tokens_target,
    class_label=class_label,
    is_debug=False,
):
    """
    Labels the tokens in the source and target lists based on their alignment.

    Args:
        tokens_source (list): List of source tokens.
        tokens_target (list): List of target tokens.
        class_label (ClassLabel): ClassLabel object for mapping label integers to strings.
        is_debug (bool, optional): Flag to enable debug mode. Defaults to False.

    Returns:
        tuple: A tuple containing the label ids and labels for each token in the source list.
    """
    source = remove_leading_under_score_chr(tokens_source[1:-1])
    target = remove_leading_under_score_chr(tokens_target[1:-1])
    target = remove_leading_empty_like_str(target)
    n = len(target)
    is_target_empty = n == 1 and target[0] == "''"
    w = 1 if n == 1 else max(int(n * 0.2), 2)

    i = find_sublist(source, target[:w])
    b_name = 0
    if i != -1:
        b_name += i
    if is_debug:
        print(f"{i=}, {b_name=}, {source=}, {target[:w]=}")

    i = find_sublist(source[::-1], target[-w:][::-1])
    b_end = len(source) - 1
    if i != -1:
        b_end -= i
    if is_debug:
        print(f"{i=}, {b_end=}, {source=}, {target[-w:]=}")

    # if is_debug: print(f"{w=}, {b_name=}, {b_end=}")

    label_ids = [0] * len(source)
    if not is_target_empty:
        label_ids[b_name] = 1
        for k in range(b_name + 1, b_end):
            label_ids[k] = 2
        label_ids[b_end] = 3

    label_ids = [-100] + label_ids + [-100]
    labels = [class_label.int2str(l) if l != -100 else "IGN" for l in label_ids]
    return label_ids, labels


def wrap_clean_and_offset(tokenizer: AutoTokenizer, examples):
    """
    Wraps the cleaning and offsetting process for the given examples using the provided tokenizer.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for tokenization.
        examples (dict): A dictionary containing the examples with "source" and "target" keys.

    Returns:
        dict: A dictionary containing the cleaned and offsetted examples with the following keys:
            - "input_ids": A list of input IDs for each example.
            - "attention_mask": A list of attention masks for each example.
            - "sentence": A list of original sentences.
            - "target": A list of target sentences.
            - "labels": A list of label IDs for each example.
    """
    list_sources = []
    list_targets = []
    list_label_ids = []
    list_labels = []
    list_input_ids = []
    list_attention_mask = []

    for idx, (source, target) in enumerate(zip(examples["source"], examples["target"])):
        tok_source = tokenizer(source)
        tok_target = tokenizer(target)
        label_ids, labels = label_token(
            tokenizer.convert_ids_to_tokens(tok_source.input_ids),
            tokenizer.convert_ids_to_tokens(tok_target.input_ids),
        )
        list_sources.append(source)
        list_targets.append(target)
        list_label_ids.append(label_ids)
        list_labels.append(labels)
        list_input_ids.append(tok_source.input_ids)
        list_attention_mask.append(tok_source.attention_mask)
    return {
        "input_ids": list_input_ids,
        "attention_mask": list_attention_mask,
        "sentence": list_sources,
        "target": list_targets,
        "labels": list_label_ids,
    }


def align_predictions(predictions, label_ids):
    """
    Aligns the predicted labels with the actual labels based on the given predictions and label IDs.

    Args:
        predictions (numpy.ndarray): The predicted labels.
        label_ids (numpy.ndarray): The label IDs.

    Returns:
        tuple: A tuple containing two lists - the aligned predicted labels and the aligned actual labels.
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    labels_list, preds_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            # 레이블 IDs = -100 무시
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
                example_preds.append(index2tag[preds[batch_idx][seq_idx]])

        labels_list.append(example_labels)
        preds_list.append(example_preds)

    return preds_list, labels_list


def compute_metrics(eval_pred):
    """
    Compute the evaluation metrics for the model predictions.

    Args:
        eval_pred (EvalPrediction): The evaluation predictions.

    Returns:
        dict: A dictionary containing the computed metrics.
            - "f1": The F1 score of the predictions.
    """
    y_pred, y_true = align_predictions(eval_pred.predictions, eval_pred.label_ids)
    return {"f1": f1_score(y_true, y_pred)}


###################
def train():
    """
    Function to train a model for token classification.

    This function performs the following steps:
    1. Parses the command line arguments or a JSON file to get the model and training arguments.
    2. Loads the tokenizer.
    3. Reads the annotated dataset from a pickle file into a dataset.
    4. Preprocesses the dataset and covert its data into token classification format (NER).
    5. Splits the dataset into training and testing sets.
    6. Fine-tunes the model.
    7. Trains the model using the Trainer class.
    8. Saves the trained model.
    """

    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(model_args.pretrained_model_name)

    df = pd.read_pickle(model_args.dataset_path)
    ds = Dataset.from_pandas(df)
    ds = ds.map(
        lambda x: wrap_clean_and_offset(tokenizer, x),
        batched=True,
        remove_columns=["source", "target"],
    )
    ds = ds.train_test_split(test_size=0.1, seed=42)

    # fine-tuning

    model_config = AutoConfig.from_pretrained(
        model_args.pretrained_model_name,
        num_labels=class_label.num_classes,
        id2label=index2tag,
        label2id=tag2index,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.pretrained_model_name, config=model_config
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    logging_steps = len(ds["train"]) // training_args.per_device_train_batch_size
    training_args.logging_steps = logging_steps

    trainer = Trainer(
        model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
    )

    logging.info(f"Start Fine Tuning... with {model_args.pretrained_model_name}")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    logging.info("Fine Tuning is done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train()

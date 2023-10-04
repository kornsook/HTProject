from torch.utils.data import random_split
from pathlib import Path
from pandas import read_csv, concat
from tqdm import tqdm
from datasets import Dataset, DatasetDict
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class HTRiskDataset(Dataset):
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)

        positive_examples_df = read_csv(data_dir / "htrisk_positive_samples.csv")
        negative_examples_df = read_csv(data_dir / "htrisk_negative_samples.csv")
        self.pos_len = len(positive_examples_df)
        self.examples = concat([positive_examples_df, negative_examples_df])

    def __getitem__(self, i: int):
        return {
            "text": self.examples.iloc[i].post,
            "label": int(i < self.pos_len),
        }

    def __len__(self):
        return len(self.examples)


def split_by_labels(dataset: Dataset) -> dict:
    result = {}

    for i in tqdm(range(len(dataset))):
        if dataset[i]["label"] not in result:
            result[dataset[i]["label"]] = []

        result[dataset[i]["label"]].append(dataset[i])

    return result


def get_dataset(
    data_dir: str, train_percent: float, val_percent: float, test_percent: float
) -> DatasetDict:
    """
    Params:
        data_dir: path to the data directory
        train_percent: percentage of the dataset to use for training. The range should be [0, 1]
        val_percent: percentage of the dataset to use for validation. The range should be [0, 1]
        test_percent: percentage of the dataset to use for testing. The range should be [0, 1]
    Note: train_percent + val_percent + test_percent should be 1
    """
    if (
        train_percent < 0
        or val_percent < 0
        or test_percent < 0
        or train_percent > 1
        or val_percent > 1
        or test_percent > 1
    ):
        raise ValueError(
            "train_percent, val_percent, and test_percent should >= 0 and <= 1"
        )
    if train_percent + val_percent + test_percent != 1:
        raise ValueError(
            "train_percent + val_percent + test_percent should be 1"
        )
    dataset = HTRiskDataset(data_dir)

    dataset_by_labels = split_by_labels(dataset)

    train_dataset = []
    val_dataset = []
    test_dataset = []

    for key in dataset_by_labels.keys():
        current_train, current_val, current_test = random_split(
            dataset_by_labels[key], [train_percent, val_percent, test_percent]
        )
        train_dataset += current_train
        val_dataset += current_val
        test_dataset += current_test

    return DatasetDict(
        {
            "train": Dataset.from_list(train_dataset),
            "validation": Dataset.from_list(val_dataset),
            "test": Dataset.from_list(test_dataset),
        }
    )

def get_dataset_json(
    data_dir: str, train_percent: float, val_percent: float
) -> DatasetDict:
    dataset = DatasetDict.from_json(
        {
            "train": os.path.join(data_dir , "train.json"),
            "test": os.path.join(data_dir , "test.json")
        }
    )
    # Convert the dictionary-like variable into a DataFrame
    df = pd.DataFrame(dataset["train"])

    # Split positive and negative samples
    positive_samples = df[df['label'] == 1]
    negative_samples = df[df['label'] == 0]
    
    # Split positive and negative samples into training and validation sets
    positive_train, positive_val = train_test_split(positive_samples, test_size=val_percent, random_state=42)
    negative_train, negative_val = train_test_split(negative_samples, test_size=val_percent, random_state=42)

    # Combine positive and negative splits to create training and validation sets
    train_data = pd.concat([positive_train, negative_train])
    val_data = pd.concat([positive_val, negative_val])
    
#     train_data, val_data = train_test_split(dataset["train"], test_size=val_percent, random_state=42)
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_data).remove_columns(['__index_level_0__']),
            "validation": Dataset.from_pandas(val_data).remove_columns(['__index_level_0__']),
            "test": dataset["test"]
        }
    )


def get_tokenizer_dataset(dataset: DatasetDict, tokenizer, task="htrisk"):
    if(task == "relatedness"):
        return dataset.map(lambda x: tokenizer(x["p1"], x["p2"], truncation=True), batched=True)
    else:
        return dataset.map(lambda x: tokenizer(x["text"], truncation=True), batched=True)

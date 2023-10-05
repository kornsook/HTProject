from datasets import DatasetDict, Dataset
import pandas as pd
from pathlib import Path

path = "data/htrp"
dataset = DatasetDict.from_json(
    {
        "train": f"{path}/train.json",
        "test": f"{path}/test.json"
    }
)
df = dataset['train'].to_pandas()
pos_df = df[df["label"] == 1]
neg_df = df[df["label"] == 0]
i = 0
while(len(pos_df) <= len(neg_df)):
    sampled_neg_df = neg_df.sample(n=len(pos_df))
    neg_df = neg_df.drop(sampled_neg_df.index)
    sub_dataset_df = pd.concat([sampled_neg_df, pos_df], axis = 0)
    sub_dataset_df = sub_dataset_df.sample(frac=1)
    sub_dataset = Dataset.from_pandas(sub_dataset_df).remove_columns(['__index_level_0__']) 
    Path(f"{path}_{i}").mkdir(parents=True, exist_ok=True)
    sub_dataset.to_json(f"{path}_{i}/train.json")
    dataset["test"].to_json(f"{path}_{i}/test.json")
    i += 1
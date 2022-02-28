from datasets import Dataset, DatasetDict
import tempfile
import requests as r
import os
import pandas as pd
from huggingface_hub import HfApi, HfFolder


dataset_id = "germeval18"
remote_test_file = "https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.test.txt"
remote_train_file = "https://raw.githubusercontent.com/uds-lsv/GermEval-2018-Data/master/germeval2018.training.txt"


def base_dataset():
    # create dataset
    train_df = pd.read_csv(remote_train_file, sep="\t", names=["text", "binary", "multi"])
    test_df = pd.read_csv(remote_test_file, sep="\t", names=["text", "binary", "multi"])
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # push to hub
    api = HfApi()

    user = api.whoami(HfFolder.get_token())
    dataset_repo_id = f"{user['name']}/germeval18"
    dataset.push_to_hub(dataset_repo_id)


def t5_dataset():
    # create dataset
    train_df = pd.read_csv(remote_train_file, sep="\t", names=["INPUT", "binary", "TARGET"])
    test_df = pd.read_csv(remote_test_file, sep="\t", names=["INPUT", "binary", "TARGET"])
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    def add_prefix(example):
        example["INPUT"] = "sentiment review: " + example["INPUT"]
        return example

    dataset = dataset.map(add_prefix)

    # push to hub
    api = HfApi()

    user = api.whoami(HfFolder.get_token())
    dataset_repo_id = f"{user['name']}/germeval18"
    dataset.push_to_hub(dataset_repo_id)


if __name__ == "__main__":
    base_dataset()
    # t5_dataset()

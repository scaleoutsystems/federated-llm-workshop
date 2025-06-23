import os
import torch
from datasets import load_dataset


dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

def load_data(data_path, is_train=True):
    """Load data from disk.

    :param data_path: Path to data file.
    :type data_path: str
    :param is_train: Whether to load training or test data.
    :type is_train: bool
    :return: Tuple of data and labels.
    :rtype: tuple
    """
    if data_path is None:
        data_path = os.environ.get("FEDN_DATA_PATH", abs_path + "/data/clients/1/hf_dataset.pt")
    print("loading from datapath ", data_path)

    if is_train:
        local_train_dataset = torch.load(data_path, weights_only=False)
        return local_train_dataset


def data_split(dataset, num_splits):
    """Split a Hugging Face Dataset into *num_splits*"""
    shards = []
    for i in range(num_splits):
        shard_i = dataset.shard(num_shards=num_splits, index=i)
        shards.append(shard_i)
    return shards

def save_data(out_dir="data"):
    dataset = load_dataset("NIH-CARD/CARDBiomedBench")
    categories = [
        "Drug Gene Relations",
        "Pharmacology",
        "Drug Meta",
        "SNP Disease Relations",
        "SMR Gene Disease Relations"
    ]

    train_datasets = []
    for category in categories:
        filtered_train_ds = dataset["train"].filter(lambda x: x["bio_category"] == category)
        filtered_train_ds = filtered_train_ds.shuffle(seed=42).select(range(150))
        train_datasets.append(filtered_train_ds)

    # make dir
    out_dir = "data"
    if not os.path.exists(f"{out_dir}/clients"):
        os.makedirs(f"{out_dir}/clients")

    for i in range(len(train_datasets)):
        subdir = f"{out_dir}/clients/{str(i + 1)}"
        os.makedirs(subdir, exist_ok=True)
        torch.save(train_datasets[i], f"{subdir}/hf_dataset.pt")


if __name__ == "__main__":
    # Prepare data if not already done
    if not os.path.exists(abs_path + "/data/clients/1"):
        save_data()
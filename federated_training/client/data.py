import os

import torch
from datasets import load_dataset

dir_path = os.path.dirname(os.path.realpath(__file__))
abs_path = os.path.abspath(dir_path)

OUT_DIR = os.path.join(abs_path, "data")


def load_data(data_path=None, is_train=True):
    if data_path is None:
        data_path = os.environ.get(
            "SCALEOUT_DATA_PATH",
            os.path.join(OUT_DIR, "clients", "1", "hf_dataset.pt"),
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            "Data either did not download, or SCALEOUT_DATA_PATH is set to a wrong path."
        )

    if is_train:
        local_train_dataset = torch.load(data_path, weights_only=False)
        return local_train_dataset


def save_data(out_dir=OUT_DIR):
    dataset = load_dataset("NIH-CARD/CARDBiomedBench")

    categories = [
        "Drug Gene Relations",
        "Pharmacology",
        "Drug Meta",
        "SNP Disease Relations",
        "SMR Gene Disease Relations",
    ]

    train_datasets = []
    for category in categories:
        filtered_train_ds = dataset["train"].filter(lambda x: x["bio_category"] == category)
        filtered_train_ds = filtered_train_ds.shuffle(seed=42).select(range(150))
        train_datasets.append(filtered_train_ds)

    os.makedirs(os.path.join(out_dir, "clients"), exist_ok=True)

    for i in range(len(train_datasets)):
        subdir = os.path.join(out_dir, "clients", str(i + 1))
        os.makedirs(subdir, exist_ok=True)
        torch.save(train_datasets[i], os.path.join(subdir, "hf_dataset.pt"))


def prepare_data():
    if not os.path.exists(os.path.join(OUT_DIR, "clients", "1")):
        save_data()


if __name__ == "__main__":
    prepare_data()

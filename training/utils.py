import os

DATASET_PATH = "dataset/real_vs_fake/real-vs-fake"

def check_dataset():
    if not os.path.exists(DATASET_PATH):
        raise Exception("Dataset not found. Follow dataset/README_dataset.md instructions.")

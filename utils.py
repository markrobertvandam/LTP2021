import numpy as np
import pandas as pd
import os
import fasttext
import fasttext.util
import shutil
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold


def initial_load_data(path):
    true_df = pd.read_csv(os.path.join(path, "True.csv"))
    fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))

    print("Len before dropping duplicates")
    print("True df", len(true_df))
    print("Fake df", len(fake_df))

    # Remove data with identical articles
    true_df = true_df.drop_duplicates(subset=["text"])
    fake_df = fake_df.drop_duplicates(subset=["text"])

    print("Len after dropping duplicate articles")
    print("True df", len(true_df))
    print("Fake df", len(fake_df))

    # Remove data with identical titles
    true_df = true_df.drop_duplicates(subset=["title"])
    fake_df = fake_df.drop_duplicates(subset=["title"])

    print("Len after dropping duplicate titles")
    print("True df", len(true_df))
    print("Fake df", len(fake_df))

    true_df = add_label_column(df=true_df, type=True)
    fake_df = add_label_column(df=fake_df, type=False)

    return true_df, fake_df


def add_label_column(df, type=True):
    if type:
        df["label"] = [1] * len(df)
    else:
        df["label"] = [0] * len(df)

    return df


def shuffle_df(df, seed=42):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def split_data(true_df, fake_df, train_split=0.8, val_split=0.1, test_split=0.1):
    true_df = shuffle_df(true_df)
    fake_df = shuffle_df(fake_df)

    train_real_df = true_df[: int(len(true_df) * train_split)]
    val_real_df = true_df[
        int(len(true_df) * train_split) : int(len(true_df) * (train_split + val_split))
    ]
    test_real_df = true_df[int(len(true_df) * (train_split + val_split)) :]

    train_fake_df = fake_df[: int(len(fake_df) * train_split)]
    val_fake_df = fake_df[
        int(len(fake_df) * train_split) : int(len(fake_df) * (train_split + val_split))
    ]
    test_fake_df = fake_df[int(len(fake_df) * (train_split + val_split)) :]

    train_df = train_real_df.append(train_fake_df, ignore_index=True)
    val_df = val_real_df.append(val_fake_df, ignore_index=True)
    test_df = test_real_df.append(test_fake_df, ignore_index=True)

    return shuffle_df(train_df), shuffle_df(val_df), shuffle_df(test_df)


def save_split_data(train_df, val_df, test_df, path):
    if not os.path.exists(path):
        os.makedirs(path)

    train_df.to_csv(os.path.join(path, "news_train.csv"), index=False)
    val_df.to_csv(os.path.join(path, "news_val.csv"), index=False)
    test_df.to_csv(os.path.join(path, "news_test.csv"), index=False)


def load_split_data(path):
    train_df = pd.read_csv(os.path.join(path, "news_train.csv"))
    val_df = pd.read_csv(os.path.join(path, "news_val.csv"))
    test_df = pd.read_csv(os.path.join(path, "news_test.csv"))

    return train_df, val_df, test_df


def check_download_embeddings():
    if not os.path.exists("data\\embeddings"):
        os.makedirs("data\\embeddings")

    if not os.path.isfile("data\\embeddings\\cc.en.300.bin"):
        fasttext.util.download_model("en")
        shutil.move("cc.en.300.bin", "data\\embeddings\\cc.en.300.bin")
        shutil.move("cc.en.300.bin.gz", "data\\embeddings\\cc.en.300.bin.gz")
        print("Embeddings downloaded successfully!")


def create_cross_validator(n_splits=10):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cv


def create_data_loaders(train_data, val_data, batch_size=1000):
    if val_data is None:
        train_data_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=False
        )
        return train_data_loader, None
    else:
        train_data_loader = DataLoader(
            dataset=train_data, batch_size=batch_size, shuffle=False
        )
        val_data_loader = DataLoader(dataset=val_data, batch_size=100, shuffle=False)
        return train_data_loader, val_data_loader


def concat_x_y(X, y):
    return list(zip(torch.from_numpy(np.array(X)), torch.from_numpy(np.array(y))))

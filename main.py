import argparse
import numpy as np

from pathlib import Path
from utils import (
    check_download_embeddings,
    initial_load_data,
    split_data,
    save_split_data,
)
from preprocessing import ArticleEmbeddings

parser = argparse.ArgumentParser(description="Run the Fake News Classifier")
parser.add_argument("data", type=Path, help="Input files directory")
parser.add_argument(
    "type", type=str, help="Type of model. Possible: text or title", default="text"
)
parser.add_argument(
    "embeddings", type=Path, help="Embeddings directory for fast text embeddings"
)
parser.add_argument(
    "save_path", type=Path, help="Path to save the preprocessed files which are fed to the model")
#parser.add_argument("--train_val_test", type=Path, help="Path to train val test splits")


def preprocess(train_df, val_df, test_df, type_of_model, save_path):
    ### Process data logics and methods

    article_embeddings = ArticleEmbeddings(ft_path="data\\embeddings\\cc.en.300.bin")
    if type_of_model == "text":
        res_train_embeddings = article_embeddings.text_embeddings(train_df)
        res_val_embeddings = article_embeddings.text_embeddings(val_df)
        res_test_embeddings = article_embeddings.text_embeddings(test_df)
    elif type_of_model == "title":
        res_train_embeddings = article_embeddings.title_embeddings(train_df)
        res_val_embeddings = article_embeddings.title_embeddings(val_df)
        res_test_embeddings = article_embeddings.title_embeddings(test_df)

    with open(save_path) as out_file:
        np.savez(
            out_file,
            X_train=res_test_embeddings,
            y_train=train_df["label"],
            X_val=res_test_embeddings,
            y_val=val_df["label"],
            X_test=res_test_embeddings,
            y_test=test_df["label"],
        )

def run_model():
    pass


def main():
    args = parser.parse_args()
    ### arguments for train_title or train_text?
    check_download_embeddings()
    #if args.train_val_test:
    #    train_df, val_df, test_df = load_split_data(args.train_val_test)

    #else:
    true_df, fake_df = initial_load_data(path=args.data)
    train_df, val_df, test_df = split_data(true_df, fake_df)
    save_split_data(train_df, val_df, test_df, "data\\splitted")

    preprocess(train_df, val_df, test_df, args.type, args.save_path)


if __name__ == "__main__":
    main()

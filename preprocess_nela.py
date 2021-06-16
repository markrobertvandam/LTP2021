import math
from typing import Tuple
import numpy as np
import pandas as pd
from tap import Tap
from fasttext import load_model
from pathlib import Path
from spacy import load
from utils import (
    load_split_data,
    check_download_embeddings,
)


class ArgumentParser(Tap):
    embedding_type: str = "title"  # type of embedding, 'text' or 'title'
    data_dir: Path = Path("data/nela_split")  # location of split nela dataframes
    save_path: Path = Path(
        "data/title_nela_embeddings.npz"
    )  # save location of created embeddings


class ArticleEmbeddings:
    def __init__(self, ft_path: Path) -> None:
        self.ft = load_model(ft_path)
        self.sp = load("en_core_web_sm")

    @staticmethod
    def cleanup_title(title: str) -> str:
        return title.replace("\n", "").replace("\r", "").replace("\t", " ")

    @staticmethod
    def __report_progress(i: int) -> None:
        if i % 1000 == 0:
            print(f"\n{i}", end="")
        elif i % 100 == 0:
            print(".", end="")

    def text_embedding(self, text: str) -> np.ndarray:
        """
        Get the fasttext embedding for the article
        as the mean of the sentence embeddings
        """
        art = self.sp(text)
        assert art.has_annotation("SENT_START")
        return np.mean(
            [
                self.ft.get_sentence_vector(self.cleanup_title(s.text))
                for s in art.sents
            ],
            axis=0,
        )

    def text_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create a numpy array of the text embeddings
        of the "text" column in the input dataframe
        """
        x = []
        y = []
        for i in range(len(df["text"])):
            self.__report_progress(i)
            if isinstance(df["text"][i], str):
                x.append(self.text_embedding(df["text"][i]))
                y.append(df["label"][i])

        print()

        return np.array(x), np.array(y, dtype=np.uint8)

    def title_embeddings(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a numpy array of the embeddings using
        the "title" column in the input dataframe
        """
        x = []
        y = []
        for i in range(len(df["title"])):
            self.__report_progress(i)

            if isinstance(df["title"][i], str):
                x.append(
                    self.ft.get_sentence_vector(
                        self.cleanup_title(df["title"][i]).lower()
                    )
                )
                y.append(df["label"][i])

        print()

        return np.array(x), np.array(y, dtype=np.uint8)


def main():
    ap = ArgumentParser()
    args = ap.parse_args()

    train_df, val_df, test_df = load_split_data(str(args.data_dir.resolve()))

    check_download_embeddings()
    article_embeddings = ArticleEmbeddings("data/embeddings/cc.en.300.bin")
    if args.embedding_type == "text":
        func = article_embeddings.text_embeddings
    elif args.embedding_type == "title":
        func = article_embeddings.title_embeddings
    else:
        print("Please select 'text' or 'title' embeddings")
        exit()

    (train_embs, train_y), (val_embs, val_y), (test_embs, test_y) = (
        func(df) for df in [train_df, val_df, test_df]
    )

    print(train_y.dtype)

    np.savez(
        args.save_path,
        X_train=train_embs,
        y_train=train_y,
        X_val=val_embs,
        y_val=val_y,
        X_test=test_embs,
        y_test=test_y,
    )


if __name__ == "__main__":
    main()

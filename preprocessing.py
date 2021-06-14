import numpy as np
import pandas as pd
from tap import Tap
from fasttext import load_model
from pathlib import Path
from spacy import load
from utils import initial_load_data, split_data, check_download_embeddings


class ArgumentParser(Tap):
    data_dir: str = "data"  # Directory that holds True.csv and Fake.csv
    embedding_type: str = "text"  # Embedding type, 'text' or 'title'
    save_path: Path = Path("data/embeddings.npz")  # Path to save the embeddings to


class ArticleEmbeddings:
    def __init__(self, ft_path: Path) -> None:
        self.ft = load_model(ft_path)
        self.sp = load("en_core_web_sm")

    @staticmethod
    def strip_reuters(text: str) -> str:
        """
        Strip "<CITY> (Reuters) - " from start of article to prevent the
        system from recognizing articles with this prefix as real news,
        since all instances of real news are from Reuters.
        """
        idx = text.find("(Reuters) - ")
        return text[12 + idx :] if idx != -1 else text

    @staticmethod
    def slashed_data(title: str) -> str:
        """
        Deals with things like \n or so in the titles
        """
        return title.replace("\n", r"").replace("\t", r"")

    def text_embedding(self, text: str) -> np.ndarray:
        """
        Get the fasttext embedding for the article
        as the mean of the sentence embeddings
        """
        art = self.sp(text)
        assert art.has_annotation("SENT_START")
        return np.mean([self.ft.get_sentence_vector(s.text) for s in art.sents], axis=0)

    def text_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        out = []
        for i in range(len(df["text"])):
            if i % 1000 == 0:
                print(f"\n{i}", end="")
            elif i % 100 == 0:
                print(".", end="")
            out.append(self.text_embedding(self.strip_reuters(df["text"][i])))
        print()
        return np.array(out)

    def title_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                self.ft.get_sentence_vector(self.slashed_data(title))
                for title in df["title"]
            ]
        )


if __name__ == "__main__":
    ap = ArgumentParser()
    args = ap.parse_args()
    true_df, fake_df = initial_load_data(args.data_dir)
    train_df, val_df, test_df = split_data(true_df, fake_df)

    check_download_embeddings()
    article_embeddings = ArticleEmbeddings(ft_path="data\\embeddings\\cc.en.300.bin")
    if args.embedding_type == "text":
        res_train_embeddings = article_embeddings.text_embeddings(train_df)
        res_val_embeddings = article_embeddings.text_embeddings(val_df)
        res_test_embeddings = article_embeddings.text_embeddings(test_df)
    elif args.embedding_type == "title":
        res_train_embeddings = article_embeddings.title_embeddings(train_df)
        res_val_embeddings = article_embeddings.title_embeddings(val_df)
        res_test_embeddings = article_embeddings.title_embeddings(test_df)

    np.savez(
        args.save_path,
        X_train=res_train_embeddings,
        y_train=train_df["label"],
        X_val=res_val_embeddings,
        y_val=val_df["label"],
        X_test=res_test_embeddings,
        y_test=test_df["label"],
    )

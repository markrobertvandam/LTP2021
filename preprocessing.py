import numpy as np
import pandas as pd
from fasttext import load_model
from pathlib import Path
from spacy import load


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
        return np.array([self.ft.get_sentence_vector(title) for title in df["title"]])

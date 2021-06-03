import argparse
from pathlib import Path
from utils import initial_load_data, load_split_data, split_data, save_split_data

parser = argparse.ArgumentParser(description="Run the Fake News Classifier")
parser.add_argument("data", type=Path, help="Input files directory")
parser.add_argument("--embeddings", type=Path, help="Embeddings directory if it exists")
parser.add_argument("--train_val_test", type=Path, help="Path to train val test splits")


def preprocess(train_df, val_df, test_df):
    ### Process data logics and methods
    pass


def run_model():
    pass


def main():
    args = parser.parse_args()
    ### arguments for train_title or train_text or both?

    ### make fasttext embeddings logic (fasttext normal library)

    if args.train_val_test:
        train_df, val_df, test_df = load_split_data(args.train_val_test)

    else:
        true_df, fake_df = initial_load_data(path=args.data)
        train_df, val_df, test_df = split_data(true_df, fake_df)
        save_split_data(train_df, val_df, test_df, "data\\splitted")


if __name__ == "__main__":
    main()

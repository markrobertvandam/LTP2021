import json
import pandas as pd
from tap import Tap
from pathlib import Path
from utils import save_split_data, split_data


class ArgumentParser(Tap):
    nela_source_dir: Path = Path("data/nela")  # NELA json source files
    nela_label_file: Path = Path("data/nela_labels.csv")  # NELA labels
    save_dir: Path = Path("data/nela_split")  # save location


def main():
    ap = ArgumentParser()
    args = ap.parse_args()

    true_df = pd.DataFrame(columns=["title", "text", "label"])
    fake_df = pd.DataFrame(columns=["title", "text", "label"])

    nela_labels = pd.read_csv(args.nela_label_file)

    for source_file in args.nela_source_dir.glob("*.json"):
        print(f"Reading {source_file.stem} articles")
        with open(source_file, "r") as f:
            source_row = nela_labels.loc[nela_labels["source"] == source_file.stem]
            if len(source_row) == 0:
                print(f"Source {source_file.stem} has no label, skipping")
                continue
            source_label = source_row.reset_index().at[0, "aggregated_label"]

            if source_label == 1:
                print(f"Source {source_file.stem} is mixed, skipping")
                continue
            source = json.load(f)

            df = true_df if source_label == 0 else fake_df
            print(
                f"Adding {source_file.stem} articles to "
                f"{'true' if source_label == 0 else 'fake'} dataframe"
            )
            for article in source:
                df.append([article["title"], article["content"], source_label])

    train_df, val_df, test_df = split_data(true_df, fake_df)
    save_split_data(train_df, val_df, test_df, str(args.save_dir.resolve()))


if __name__ == "__main__":
    main()

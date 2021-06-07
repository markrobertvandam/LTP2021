import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("true", help="true file")
parser.add_argument("fake", help="fake file")

def load_data(args):
    true = pd.read_csv(args.true)
    fake = pd.read_csv(args.fake)

    return true, fake

def check_data(df, t = 'true'):
    print("Rows in {}: {}".format(t, len(df.index)))
    df = df.drop_duplicates(subset=['text'])
    print("Rows after dropping text: {}".format(len(df.index)))
    df = df.drop_duplicates(subset=['title'])
    print("Rows after dropping title: {}".format(len(df.index)))


def main():
    args = parser.parse_args()
    true, fake = load_data(args)
    check_data(true, t='true')
    print("-----------------------")
    check_data(fake, t='fake')

if __name__ == "__main__":
    main()
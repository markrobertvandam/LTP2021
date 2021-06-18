# LTP2021
Language Technology Project 2021

## Run with first dataset

- Download the dataset from https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
- Run preprocessing.py with the data folder and text/title argument depending on which you want
- Run model.py with the npz file from the previous step (save output if you want to run eval.py)

## Run with second dataset

- Download the JSON file and labels files from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O7FWPO
- Delete the tweet json as we only use news articles. For folder structure see split_nela.py args
- Run split_nela.py with arguments as described in the python file (or use the described folder structure)
- Run preprocess_nela.py with resulting files from the previous step
- Run model.py with the npz file from the previous step (save output if you want to run eval.py)

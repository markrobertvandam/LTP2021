import pandas as pd
import os

def initial_load_data(path):
    true_df = pd.read_csv(os.path.join(path, "True.csv"))
    fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))
    return true_df, fake_df
    
def split_data(true_df, fake_df, train_split=0.8, val_split=0.1, test_split=0.1):
    train_real_df = true_df[:int(len(true_df)*train_split)]
    val_real_df = true_df[int(len(true_df)*train_split):int(len(true_df)*(train_split+val_split))]
    test_real_df = true_df[int(len(true_df)*(train_split+val_split)):]
    
    train_fake_df = fake_df[:int(len(fake_df)*train_split)]
    val_fake_df = fake_df[int(len(fake_df)*train_split):int(len(fake_df)*(train_split+val_split))]
    test_fake_df = fake_df[int(len(fake_df)*(train_split+val_split)):]
    
    train_df = train_real_df.append(train_fake_df, ignore_index=True)
    val_df = val_real_df.append(val_fake_df, ignore_index=True)
    test_df = test_real_df.append(test_fake_df, ignore_index=True)
    
    return train_df, val_df, test_df
    
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
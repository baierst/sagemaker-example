import pandas as pd
import numpy as np

df = pd.read_csv("data/ratings.csv")

msk = np.random.rand(len(df)) < 0.8

df_train = df[msk]
df_val = df[~msk]

df_train.to_csv("data/training.csv", index=False)
df_val.to_csv("data/validation.csv", index=False)
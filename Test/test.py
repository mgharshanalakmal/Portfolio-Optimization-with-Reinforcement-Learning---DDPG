import pandas as pd
import numpy as np
import os

# import tensorflow as tf
# from tf_agents.environments import py_environment


df = pd.read_csv("..\Data\ACL Historical Data.csv").dropna()
df["Volume"] = df["Vol."].str[:-1].astype(float)
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")


def convert_volume(vol):
    value = float(vol[:-1])
    unit = vol[-1]
    if unit == "K":
        return value * 1000
    elif unit == "M":
        return value * 1000000
    else:
        return value


df["Volume"] = df["Vol."].apply(convert_volume)
print(df.head(2))

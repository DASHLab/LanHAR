import pandas as pd
import numpy as np
import random
import ast
import re
from tqdm import tqdm
from data_processing import *

    

def read_data(source, target):

    source_text, source_data = load_dataset(f"data/df_{source}.csv")
    target_text, target_data = load_dataset(f"data/df_{target}.csv")

    return source_text, source_data, target_text, target_data


def read_data_stage1(source):
    source_text, source_data = load_dataset(f"data/df_{source}.csv")

    return source_text, source_data

    
def generate_step1(target_text,target_data,target):

    label_ = ['walk', 'still', 'stairsdown', 'stairsup']

    data = []
    for i in range(target_text.shape[0]):
        text_label = target_text["label"].iloc[i]
        text1a, text1b = str(target_text["pattern"].iloc[i]), str(
                target_text["label"].iloc[i])
        time_series = target_data[i]
        label = label_.index(text_label)

        if text_label == 'walk':
            index_j = random.choice([0])
            com_s = target_text[target_text["label"] == label_[index_j]]
            com_s = com_s.sample(n=1)
            com_s_a, com_s_b = str(com_s["pattern"].iloc[0]), str(
                com_s["label"].iloc[0])

            index_j = random.choice([1,2,3])
            com_ds = target_text[target_text["label"] == label_[index_j]]
            com_ds = com_ds.sample(n=1)
            com_ds_a, com_ds_b = str(com_ds["pattern"].iloc[0]), str(
                com_ds["label"].iloc[0])
            data.append((time_series, text1a, text1b, label, com_s_a, com_s_b, com_ds_a, com_ds_b))

        if text_label == 'still':
            index_j = random.choice([1])
            com_s = target_text[target_text["label"] == label_[index_j]]
            com_s = com_s.sample(n=1)
            com_s_a, com_s_b = str(com_s["pattern"].iloc[0]), str(
                com_s["label"].iloc[0])

            index_j = random.choice([0, 2, 3])
            com_ds = target_text[target_text["label"] == label_[index_j]]
            com_ds = com_ds.sample(n=1)
            com_ds_a, com_ds_b = str(com_ds["pattern"].iloc[0]), str(
                com_ds["label"].iloc[0])
        data.append((time_series, text1a, text1b, label, com_s_a, com_s_b, com_ds_a, com_ds_b))

        if text_label == 'stairsdown' or text_label == 'stairsup':
            index_j = random.choice([2,3])
            com_s = target_text[target_text["label"] == label_[index_j]]
            com_s = com_s.sample(n=1)
            com_s_a, com_s_b = str(com_s["pattern"].iloc[0]), str(
                com_s["label"].iloc[0])

            index_j = random.choice([0, 1])
            com_ds = target_text[target_text["label"] == label_[index_j]]
            com_ds = com_ds.sample(n=1)
            com_ds_a, com_ds_b = str(com_ds["pattern"].iloc[0]), str(
                com_ds["label"].iloc[0])
        data.append((time_series, text1a, text1b, label, com_s_a, com_s_b, com_ds_a, com_ds_b))

    return data


def generate_step2(target_text,target_data,source_text,source_data):
    data = []
    label_ = ['walk', 'still', 'stairsdown', 'stairsup']
    for i in range(int(target_text.shape[0])):
        text1a, text1b = str(target_text["pattern"].iloc[i]), str(
            target_text["label"].iloc[i])
        time_series = target_data[target_text.index[i]]
        label = label_.index(target_text["label"].iloc[i])
        data.append((time_series, text1a, text1b, label))
    for i in range(source_text.shape[0]):
        text1a, text1b = str(source_text["pattern"].iloc[i]), str(
            source_text["label"].iloc[i])
        time_series = source_data[source_text.index[i]]
        label = label_.index(source_text["label"].iloc[i])
        data.append((time_series, text1a, text1b, label))
    return data


def generate_step3(target_text, target_data):
    data = []
    bb = target_text[target_text["label"].isin(['walk', 'still', 'stairsdown', 'stairsup'])]
    label_ = ['walk', 'still', 'stairsdown', 'stairsup']
    for i in range(bb.shape[0]):
        text1a, text1b = str(bb["pattern"].iloc[i]), str(
            bb["label"].iloc[i])
        time_series = target_data[bb.index[i]]
        label = label_.index(bb["label"].iloc[i])
        data.append((time_series, text1a, text1b, label))
    return data





def _rotate_gyro_to_aligned(gyro_xyz, R):
    gyro_xyz = np.asarray(gyro_xyz, dtype=float)
    return gyro_xyz @ R.T 

def parse_acc_string(s: str) -> np.ndarray:
    numbers = np.fromstring(s.replace("\n", " ").replace("[", " ").replace("]", " "), sep=" ")
    return numbers.reshape(-1, 3)


def load_dataset(csv_path, label_replace=True):
    df = pd.read_csv(csv_path)
    if label_replace:
        df.loc[df["label"] == 'stand', 'label'] = "still"
        df.loc[df["label"] == 'sit', 'label'] = "still"
    df["acc_matrix"] = df["acc"].apply(parse_acc_string)
    df["gyro_matrix"] = df["gyro"].apply(parse_acc_string)

    all_accs, all_gyros = [], []
    for acc, gyro in zip(df["acc_matrix"], df["gyro_matrix"]):
        pre = preprocess_acc_segment(acc, fs=50.0, mode="auto")
        acc_raw = pre["aligned"]["acc_xyz"]
        gyro_raw = _rotate_gyro_to_aligned(gyro, pre["gravity"]["rot_R"])
        all_accs.append(acc_raw)
        all_gyros.append(gyro_raw)

    data = np.concatenate([np.stack(all_accs), np.stack(all_gyros)], axis=2)
    return df, data
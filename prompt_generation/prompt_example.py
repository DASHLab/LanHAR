import numpy as np
import pandas as pd
from prompt import *

if __name__ == "__main__":
    
    data = np.load("data_20_120.npy")
    acc  = data[0, :, :3].astype(float)   
    gyro = data[0, :, 3:6].astype(float)   
    dataset_name = "uci"
    prompt = generate_promt(acc, gyro, dataset_name, '')
    print(prompt)

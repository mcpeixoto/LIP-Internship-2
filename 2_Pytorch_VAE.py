import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torch.optim import Adam
from pytorch_lightning import Trainer
import os
import numpy as np
import pandas as pd
from os.path import join
from typing import Optional
from config import processed_data_path
from sklearn.utils import shuffle

class dataset(Dataset):
    def __init__(self, key, random_seed=42):
        # Check if key is valid
        assert key in {'VLQ_HG', 'VLQ_SEM_HG', 'bkg', 'FCNC'}, "Invalid Key!"

        file = join(processed_data_path, key+".csv")

        # TODO: IMPROVE, ugly solution
        xy = pd.read_csv(file)
        xy.drop(columns=['name'], inplace=True)
        xy = shuffle(xy, random_state=random_seed)
        xy = xy.to_numpy()

        self.x = torch.from_numpy(xy)
        self.y = torch.from_numpy(np.ones(shape=(xy.shape[0], 1)))
        self.n_samples = xy.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples






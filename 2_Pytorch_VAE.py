import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torch.utils.data.dataset import TensorDataset
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

# Defining the dataset
class dataset(Dataset): #
    def __init__(self, key, type, random_seed=42):
        # TODO: Improve efficiency/how names are handled
        # Check if key is valid
        assert key in {'VLQ_HG', 'VLQ_SEM_HG', 'bkg', 'FCNC'}, "Invalid Key!"

        # With specified key, get data
        file = join(processed_data_path, key+".csv")
        data = pd.read_csv(file, index_col=0)

        # Shuffle the dataframe
        data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # This data we want on a seperate variable
        self.names = data["name"]
        self.weights = torch.from_numpy(data["weights"].to_numpy())

        data.drop(["name", "weights"], axis=1, inplace=True)
        
        # In order to have std = 1 and mean = 0
        data = (data-data.mean())/data.std()

        # This will equally devide the dataset into 
        # train, validation and test
        train, validation, test = np.split(data.sample(frac=1), [int(len(data)*(1/3)), int(len(data)*(2/3))])
        
        if type == "train":
            data = train
        elif type == "validation":
            data = validation
        elif type == "test":
            data = test

        self.data = torch.from_numpy(data.to_numpy())
        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.weights[index], self.names[index]

    def __len__(self):
        return self.n_samples






class VAE(pl.LightningModule):
    def __init__(self, hidden_size, alpha, lr, dataset):
        """
        Args:
        - > hidden_size (int): Latent Hidden Size
        - > alpha (int): Hyperparameter to control the importance of
        reconstruction loss vs KL-Divergence Loss
        - > lr (float): Learning Rate, will not be used if auto_lr_find is used.
        - > dataset (Optional[str]): Dataset to used
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.lr = lr
        self.alpha = alpha
        self.dataset = dataset

        # Architecture
        self.encoder = nn.Sequential(
            nn.Linear(71, 128), 
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(), 
            nn.Linear(128, hidden_size),
            nn.LeakyReLU()
        )

        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2sigma = nn.Linear(hidden_size, hidden_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128), 
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 71), 
            nn.LeakyReLU(),
        )

    def encode(self, x):
        # Pass through encoder
        out = self.encoder(x)
        mu = self.hidden2mu(out)
        sigma = self.hidden2sigma(out)
        return mu, sigma

    def decode(self, x):
        # Pass through encoder
        return self.decoder(x)

    def reparametrize(self, mu, sigma):
        # Reparametrization Trick
        # It outputs a sample of the dist.
        # mu -> average | sigma -> std
        # e -> Epsilon. Sample from the normal distribution, same shape as mu
        e = torch.randn(size=(mu.size(0), mu.size(1))) 
        e = e.type_as(mu) # To the same device
        return mu + sigma*e

    def forward(self, x):
        # Pass through encoder
        mu, sigma = self.encode(x)
        # Reparametrization Trick
        hidden = self.reparametrize(mu, sigma)
        # Pass through decoder
        output = self.decoder(hidden)

        return mu, sigma, output

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    # Functions for dataloading
    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass





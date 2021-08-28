from IPython import get_ipython

# %%
import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from pytorch_lightning import Trainer
import os
import numpy as np
import pandas as pd
from os.path import join
from typing import Optional
from config import processed_data_path
from sklearn.utils import shuffle
from tqdm import tqdm
from optuna.integration import PyTorchLightningPruningCallback
from scipy.stats import wasserstein_distance 
import joblib
import optuna
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import r2_score
import threading, concurrent
import glob

from statsmodels.distributions.empirical_distribution import StepFunction
from scipy.stats import wasserstein_distance
import numpy as np

class wECDF(StepFunction):
    def __init__(self, x, w, side="right"):
        x = np.array(x, copy=True)
        w = np.array(w, copy=True)
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = np.cumsum(w[sorted_indices])
        super(wECDF, self).__init__(x, y, side=side, sorted=True)

def compare_continuous(x1, w1, x2, w2):
    w1 = w1 / w1.sum()
    w2 = w2 / w2.sum()
    wecdf1 = wECDF(x1, w1)
    wecdf2 = wECDF(np.clip(x2, x1.min(), x1.max()), w2)
    u1 = wecdf1.y
    u2 = wecdf2(wecdf1.x)
    return wasserstein_distance(u1, u2)


def compare_integer(x1, w1, x2, w2):
    d = np.diff(np.unique(x1)).min()
    left_of_first_bin = x1.min() - float(d) / 2
    right_of_last_bin = x1.max() + float(d) / 2
    bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)
    h1, _ = np.histogram(x1, weights=w1, bins=bins)
    h2, _ = np.histogram(np.clip(x2, x1.min(), x1.max()), weights=w2, bins=bins)
    u1 = np.cumsum(h1) / h1.sum()
    u2 = np.cumsum(h2) / h2.sum()
    return wasserstein_distance(u1, u2)
    

class _dataset(Dataset): #
    def __init__(self, variant, category, random_seed=42):
        self.variant = variant
        """ 
        variant -> 'VLQ_HG', 'VLQ_SEM_HG', 'bkg', 'FCNC'
        category -> 'train, validation', 'test', 'all'
        tensor -> if true will return the data as a tensor, if False will return as a DataFrame
        """
        # TODO: Improve efficiency/handle names
        
        # Sanity checks
        assert variant in {'bkg', 'signal'}, "Invalid variant!"
        assert category in {'train', 'validation', 'test', 'all'}, "Invalid category!"

        # With specified variant, get data
        if variant == "bkg":
            self.data = pd.read_hdf(join(processed_data_path, "bkg.h5"), index_col=0)
        elif variant == "signal":
            self.data = pd.concat(
                [pd.read_hdf(path, index_col=0) for path in glob.glob(join(processed_data_path, "[!bkg]*"))]
            )

        # Shuffle the dataframe
        self.data = self.data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        # This will equally devide the dataset into 
        # train, validation and test
        train, validation, test = np.split(self.data.sample(frac=1), [int(len(self.data)*(1/3)), int(len(self.data)*(2/3))])
        
        if category == "train":
            self.data = train
        elif category == "validation":
            self.data = validation
        elif category == "test":
            self.data = test
        else:
            pass

        del train, validation, test
        
        # This data we want on a seperate variable
        if category != "all":
            # Weights
            self.weights = self.data["weights"]

            # Name
            self.name = self.data["name"]

            self.data.drop(columns=["name", "weights"], inplace=True)

            self.n_samples = self.data.shape[0]

        if self.data.isnull().values.any():
            print("WARNING! DATA HAS NAN")

        # Everything to tensors
        
        #self.weights = torch.from_numpy(self.weights.to_numpy(dtype=np.float16))
        #self.data = torch.from_numpy(self.data.to_numpy(dtype=np.float16))


    def __getitem__(self, index):
        return torch.from_numpy(self.data.iloc[index].to_numpy(dtype=np.float16)), torch.from_numpy(np.array([self.weights.iloc[index]]))
        #return tuple(self.data.iloc[index], self.weights.iloc[index])

    def __len__(self):
        return self.n_samples

    def all_data(self):
        
        try:
            to_return = self.data
            to_return['weights'] = self.weights
            to_return['name'] = self.name 
            return to_return
        except:
            return self.data


# %%
def compare_distributions_binned_aux(x1, w1, x2, w2, bins=1000):
    EPS = np.finfo(float).eps
    h_init, b = np.histogram(x1, bins=bins, weights=w1)
    dists = [i for i in range(len(h_init))]
    h_gene, _ = np.histogram(np.clip(x2, x1.min(), x1.max()), bins=b, weights=w2)
    wd = wasserstein_distance(dists, dists, h_init + EPS, h_gene + EPS)
    return wd


def compare_distributions_binned(x1, w1, x2, w2, bins=500):
    
    hidden_size = x1.shape[1]
    batch_size = x1.shape[0]


    total_WD=0

    for i in range(hidden_size):
           total_WD += compare_distributions_binned_aux(
                                        x1[:, i], 
                                        w1[i]*np.ones(batch_size), 
                                        x2[:, i], 
                                        w2[i]*np.ones(batch_size),
                                        bins
                                        )
    return total_WD / hidden_size


# %%
class VAE(pl.LightningModule):
    def __init__(self, trial, zdim, dataset, batch_size):
        """
        Args:
        - > variant e {'VLQ_HG', 'VLQ_SEM_HG', 'bkg', 'FCNC'}; it's the type of data
        - > hidden_size : Latent Hidden Size
        - > alpha : Hyperparameter to control the importance of
        reconstruction loss vs KL-Divergence Loss
        - > lr : Learning Rate, will not be used if auto_lr_find is used.
        - > dataset : Dataset to used
        """
        super().__init__()
        self.trial = trial
        self.save_hyperparameters = True
        self.dataset = dataset
        self.batch_size = batch_size
        self.hidden_size = zdim 
        hidden_size = self.hidden_size # yes I am lazy
        self.lr = trial.suggest_float("lr", 1e-10, 1e-2, log=True)
        self.alpha = trial.suggest_int("alpha", 1, 10000, step=5)
        self.best_score = None


        #### Architecture
        #-> Encoder
        n_layers_encoder = trial.suggest_int("n_layers_encoder", 1, 20)
        layers = []

        in_features = 47
        for i in range(n_layers_encoder):
            out_features = trial.suggest_int("n_units_encoder_l{}".format(i), 10, 500, step=10)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())

            in_features = out_features

        # Ultima layer
        layers.append(nn.Linear(in_features, hidden_size))
        # layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*layers)

        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        
        #-> Decoder
        n_layers_encoder = trial.suggest_int("n_layers_decoder", 1, 20)
        layers = []

        in_features = hidden_size
        for i in range(n_layers_encoder):
            out_features = trial.suggest_int("n_units_decoder_l{}".format(i), 5, 500, step=10)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())

            in_features = out_features

        # Ultima layer
        layers.append(nn.Linear(in_features, 47))
        # layers.append(nn.LeakyReLU())

        self.decoder = nn.Sequential(*layers)


    def encode(self, x):
        # Pass through encoder
        out = self.encoder(x)
        mu = self.hidden2mu(out)
        log_var = self.hidden2log_var(out)
        return mu, log_var

    def decode(self, x):
        # Pass through encoder
        return self.decoder(x)

    def reparametrize(self, mu, log_var):
        # Reparametrization Trick
        # It outputs a sample of the dist.
        # mu -> average | log_var -> std
        
        log_var = torch.exp(0.5*log_var)
        z = torch.randn(size=(mu.size(0), mu.size(1))) # log_var, normal distribution
        z = z.type_as(mu)
        return mu + log_var*z

    def forward(self, x):
        # Pass through encoder
        mu, log_var = self.encode(x)
        # Reparametrization Trick
        hidden = self.reparametrize(mu, log_var)
        # Pass through decoder
        output = self.decoder(hidden)

        return mu, log_var, output, hidden

    def training_step(self, batch, batch_idx):
        x, weights = batch
        # Pass
        mu, log_var, x_out, _ = self.forward(x)

        # Losses
        kl_loss = (-0.5*(1+log_var - mu**2 -torch.exp(log_var)).sum(dim=1)).mean(dim=0)

        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)

        loss = recon_loss*self.alpha + kl_loss

        # Weights on final loss
        loss = (weights * loss) / weights.sum()
        loss = torch.mean(loss, dtype=torch.float32)

        if loss.isnan().any():
            raise KeyboardInterrupt
            pass

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch):
        ### WIP
        x = batch
        mu, log_var, x_out, hidden = self.forward(x)

        # Loss
        kl_loss = (-0.5*(1+log_var - mu**2 -
                         torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)
        loss = recon_loss*self.alpha + kl_loss

        return  mu, log_var, x_out, hidden

    def validation_step(self, batch, batch_idx):
     
        x, weights = batch

        _, _, output, _ = self.forward(x)

        x = x.cpu().numpy()
        weights = weights.cpu().numpy()
        output = output.cpu().numpy()

        try:
            objective_score = r2_score(x, output, sample_weight=weights)
        except:
            print("\n[-] Erro! ")
            raise KeyboardInterrupt


        self.log('objective_score', objective_score, prog_bar=True)

        if self.best_score is None:
            self.best_score = objective_score
        elif objective_score > self.best_score:
            self.best_score = objective_score
        else:
            pass

        
        self.trial.report(objective_score, batch_idx)

        if self.trial.should_prune():
            raise optuna.TrialPruned()


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    # Functions for dataloading
    def train_dataloader(self):
        train_set = _dataset(self.dataset, category="train")
        return DataLoader(train_set, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        val_set = _dataset(self.dataset, category='test')
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=12)


# %%
def objective(trial):
    global zdim

    name = f"r2_zStudy-zdim_{zdim}_trial_{trial.number}"

    logger = TensorBoardLogger("lightning_logs", name=name)

    max_epochs=100 #trial.suggest_int("max_epochs", 50, 500, step=10)
    patience=10 #trial.suggest_int("patience", 10, 200, step=10)

    trainer = pl.Trainer(
        auto_scale_batch_size=True,
        gpus=1,
        logger=logger,
        max_epochs=max_epochs,
        precision=16,
        check_val_every_n_epoch=1,
        callbacks=[
            EarlyStopping(monitor="objective_score", patience=patience, mode="max"),
            ModelCheckpoint(dirpath="models", filename=name, monitor="objective_score", mode="max")]
    )

    model = VAE(trial, zdim=zdim, dataset = "bkg", batch_size=512)

    # FIt the model
    trainer.fit(model)

    return model.best_score

def run(zdim):
    print("ZDIM = ", zdim)
    study = optuna.create_study(direction="maximize", 
                                study_name=f"zStudy - zdim {zdim} - Optimizing the VAE with R2", 
                                storage="sqlite:///optimization.db", 
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1, n_min_trials=10),
                                load_if_exists=True)


    study.optimize(objective, n_trials=100)


if __name__ == "__main__":
    zdims = [55,65,75,85,95]#[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]

    for x in zdims:
        global zdim
        zdim = x
        run(x)


"""
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(" TRIAL NUMBER:", trial.number)

    params = study.best_trial.params

    # Manually change them
    #params['hidden_size'] = 16
    params['max_epochs'] = 1000
    params['patience'] = 200


    # Name of the model
    name = f"CustomTrain_R2-Data_vs_Reconstruction_trial_{trial.number}"
    print("Name:", name)
    print(params)

    logger = TensorBoardLogger("lightning_logs", name=name)

    trainer = pl.Trainer(
            gpus=1,
            logger=logger,
            max_epochs=params['max_epochs'],
            precision=16,
            check_val_every_n_epoch=1,
            callbacks=[
                EarlyStopping(monitor="objective_score", patience=params['patience'], mode="max"),
                ModelCheckpoint(dirpath="models", filename=name, monitor="objective_score", mode="max")]
        )

    model = VAE(optuna.trial.FixedTrial(params), dataset = "bkg", batch_size=512)
    
    trainer.fit(model)
"""
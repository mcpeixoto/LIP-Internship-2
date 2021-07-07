from IPython import get_ipython

# %%
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
from tqdm.notebook import tqdm
from optuna.integration import PyTorchLightningPruningCallback
from scipy.stats import wasserstein_distance 
import joblib
import optuna
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import r2_score


class _dataset(Dataset): #
    def __init__(self, variant, category, random_seed=42):
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
        self.data = pd.read_hdf(join(processed_data_path, "data.h5"), key=variant, index_col=0)

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
            #self.name = self.data["name"]

            self.data.drop(columns=["name", "weights"], inplace=True)

            self.n_samples = self.data.shape[0]

        if self.data.isnull().values.any():
            print("WARNING! DATA HAS NAN")

        # Everything to tensors
        
        #self.weights = torch.from_numpy(self.weights.to_numpy(dtype=np.float16))
        #self.data = torch.from_numpy(self.data.to_numpy(dtype=np.float16))


    def __getitem__(self, index):
        return torch.from_numpy(self.data.iloc[index].to_numpy(dtype=np.float32)), torch.from_numpy(np.array([self.weights.iloc[index]]))
        #return tuple(self.data.iloc[index], self.weights.iloc[index])

    def __len__(self):
        return self.n_samples

    def all_data(self):
        return self.data

# %% [markdown]
# ## Defining the model

# %%
class VAE(pl.LightningModule):
    def __init__(self, trial, batch_size, dataset):
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
        self.dataset = dataset
        self.batch_size = batch_size
        self.hparams.batch_size = batch_size
        self.hidden_size = trial.suggest_int("hidden_size", 2, 40)
        hidden_size = self.hidden_size # yes I am lazy
        self.lr = trial.suggest_float("lr", 1e-8, 1e-2, log=True)
        self.alpha = trial.suggest_int("alpha", 1, 3000)
        self.best_score = None
        ## Architecture
        # Encoder
        n_layers_encoder = trial.suggest_int("n_layers_encoder", 1, 4)
        layers = []

        in_features = 75
        for i in range(n_layers_encoder):
            out_features = trial.suggest_int("n_units_encoder_l{}".format(i), 5, 256)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())

            in_features = out_features

        # Ultima layer
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*layers)

        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        
        # Decoder
        n_layers_encoder = trial.suggest_int("n_layers_decoder", 1, 4)
        layers = []

        in_features = hidden_size
        for i in range(n_layers_encoder):
            out_features = trial.suggest_int("n_units_decoder_l{}".format(i), 5, 256)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.LeakyReLU())

            in_features = out_features

        # Ultima layer
        layers.append(nn.Linear(in_features, 75))
        # layers.append(nn.LeakyReLU())

        self.decoder = nn.Sequential(*layers)

        ## Load bkg data for
        # being used at self.on_epoch_end
        #self.bkg = _dataset(category='test', variant='bkg', tensor=False).all_data().to_numpy()

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
        kl_loss = (-0.5*(1+log_var - mu**2 -
                         torch.exp(log_var)).sum(dim=1)).mean(dim=0)

        recon_loss_criterion = nn.MSELoss()
        recon_loss = recon_loss_criterion(x, x_out)

        loss = recon_loss*self.alpha + kl_loss

        # Weights on final loss
        loss = (weights * loss) / weights.sum()
        loss = torch.mean(loss, dtype=torch.float32)

        if loss.isnan().any():
            raise KeyboardInterrupt

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
        with torch.no_grad():
            x, weights = batch
            # Pass
            _, _, output, _ = self.forward(x)

            x = x.cpu().numpy()
            output = output.cpu().numpy()

            #print("Input", np.isnan(x).any())
            #print("Output", np.isnan(output).any())


            try:
                objective_score = r2_score(x,output)
            except:
                print("\n[-] Erro! ")
                # objective_score = np.inf
                raise KeyboardInterrupt

            self.log('objective_score', objective_score, prog_bar=True)

            if self.best_score is None:
                self.best_score = objective_score
            elif objective_score > self.best_score:
                self.best_score = objective_score
            else:
                pass


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

    name = "r2_trial_{}".format(trial.number)

    logger = TensorBoardLogger("lightning_logs", name=name)

    max_epochs=trial.suggest_int("max_epochs", 50, 500, step=5)
    patience=trial.suggest_int("patience", 50, 200, step=5)

    trainer = pl.Trainer(
        #move_metrics_to_cpu=True,
        auto_scale_batch_size='binsearch',
        gpus=1,
        logger=logger,
        max_epochs=max_epochs,
        precision=16,
        check_val_every_n_epoch=10,
        callbacks=[
            EarlyStopping(monitor="objective_score", patience=patience, mode="max"),
            ModelCheckpoint(dirpath="models", filename=name, monitor="objective_score", mode="max")]
    )

    model = VAE(trial, dataset = "bkg", batch_size=1)

    # Find batch size
    #trainer.tune(model)

    # Invoke method
    #new_batch_size = tuner.scale_batch_size(model, *extra_parameters_here)

    # Override old batch size (this is done automatically)
    #model.hparams.batch_size = new_batch_size

    # FIt the model
    trainer.fit(model)

    return model.best_score

# %% [markdown]
# ## Training

study = optuna.create_study(direction="maximize", study_name="Optimizing the VAE with r2", storage="sqlite:///r2-optimization.db", load_if_exists=True)
study.optimize(objective, timeout=10*60*60)#n_trials=200)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print(" TRIAL NUMBER:", trial.number)
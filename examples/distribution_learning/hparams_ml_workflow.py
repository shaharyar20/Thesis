import os
import random
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import torch
import pickle
from pathlib import Path
from datetime import datetime

from mlbm.distribution_learning.datasets.collision_2d_graph_dataset import Collision2DGraphDataset
import lightning.pytorch as pl
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from mlbm.distribution_learning.networks.loss_utilities import LossOption
from mlbm.distribution_learning.networks.activation_utilities import ActivationOption
from mlbm.distribution_learning.trial_networks.gcn1 import GraphCollisionNetwork1
from mlbm.distribution_learning.trial_networks.gcn2 import GraphCollisionNetwork2
from mlbm.distribution_learning.trial_networks.gcn3 import GraphCollisionNetwork3
from mlbm.distribution_learning.trial_networks.gcn4 import GraphCollisionNetwork4

# torch.autograd.set_detect_anomaly(True)

# torch.set_float32_matmul_precision('high')


dataset_path = './data/entropic_10k_gs2_exp3_5_u05'
model_path = Path("./models/gcn4_entropic_10k_gs2_exp3_5_u05.pth")
load_existing_model = False
train_val_test_split = {"train":0.8, "val":0.1, "test":0.1}

hyperparameter_options = {
    "emb_dim": (9, 20),
    "bias": [True, False],
    "residual_connection": [True, False],
    "pre_collision_residual": [True, False],
    "loss_function": [LossOption.l1, LossOption.mse, LossOption.msre],
    "activation": [ActivationOption.tanh, ActivationOption.gelu, ActivationOption.silu],
    "lr": [5e-3, 6e-3, 7e-3, 8e-3],#(7e-3, 9e-3),
    "weight_decay": [1e-12, 1e-10, 1e-8],
    "step_size": (5, 10),
    "gamma": (0.7, 0.9),
}

def sample_hyperparameters():
    return {
        "emb_dim": random.randint(*hyperparameter_options["emb_dim"]),
        "aggr": "add",
        "bias": False, #random.choice(hyperparameter_options["bias"]),
        "residual_connection": True, #random.choice(hyperparameter_options["residual_connection"]),
        "pre_collision_residual": True, #random.choice(hyperparameter_options["pre_collision_residual"]),
        "loss_function": LossOption.l1, #random.choice(hyperparameter_options["loss_function"]),
        "activation": ActivationOption.gelu, #random.choice(hyperparameter_options["activation"]),
        "lr": random.choice(hyperparameter_options["lr"]), #random.uniform(*hyperparameter_options["lr"]),
        "weight_decay": random.choice(hyperparameter_options["weight_decay"]),
        "step_size": random.randint(*hyperparameter_options["step_size"]),
        "gamma": random.uniform(*hyperparameter_options["gamma"]),
    }

# hparams = {
#     "emb_dim": 20,
#     "aggr": "add",
#     "bias": False,
#     "residual_connection": False,
#     "pre_collision_residual": True,
#     "loss_function": LossOption.l1,
#     "lr": 1e-4,
#     "weight_decay": 0,
#     "step_size": 2,
#     "gamma": 0.8,
# }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 140
num_trials = 30
results = []

if __name__ == "__main__":

    # dataset = CollisionGraphDataset.load(dataset_path)
    dataset = Collision2DGraphDataset.load(dataset_path)
    train_dataset, val_dataset, test_dataset = dataset.splits(train_val_test_split)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for trial in range(num_trials):
        hparams = sample_hyperparameters()
        trial_model_path = model_path.with_name(f"{model_path.stem}_trial_{trial}{model_path.suffix}") 

        # model = GraphCollisionNetwork(
        # model = LightGraphCollisionNetwork(
        # model = GraphCollisionNetwork1(
        # model = GraphCollisionNetwork2(
        #     hparams=hparams,
        #     input_dim=train_dataset[0].x.shape[1], #+ 1,  # +1 for weights
        #     edge_dim=train_dataset[0].edge_attr.shape[1] if train_dataset[0].edge_attr is not None else 0,
        #     output_dim=train_dataset[0].y.shape[1],
        #     # edge_index=train_dataset[0].edge_index,
        # )
        # model = torch.jit.script(model)
        # print(model)
        # print(device)
        model = GraphCollisionNetwork4(
            hparams=hparams,
            input_dim=train_dataset[0].x.shape[1],  # + 1,  # +1 for weights
            edge_dim=train_dataset[0].edge_attr.shape[1] if train_dataset[0].edge_attr is not None else 0,
            output_dim=train_dataset[0].y.shape[1],
        )

        logger = MLFlowLogger(experiment_name="LargerModelSmallerVelocity")

        trainer = pl.Trainer(
            callbacks=[
                # EarlyStopping(monitor="val_loss", mode="min", patience=30),
                ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_msre_loss"),
                LearningRateMonitor(logging_interval='step'),
            ],
            max_epochs=epochs,
            logger=logger,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            # log_every_n_steps=1,
            # val_check_interval=1.0,
            # check_val_every_n_epoch=50,
            enable_progress_bar=True,
            # reload_dataloaders_every_n_epochs=1,
        )

        print(f"Starting trial {trial + 1} with hyperparameters: {hparams}")
        trainer.fit(model, train_loader, val_loader)

        val_loss = trainer.callback_metrics["val_msre_loss"].item()
        results.append({"trial": trial, "hparams": hparams, "val_loss": val_loss})
        print(f"Trial {trial + 1} finished with validation loss: {val_loss}")

        trainer.save_checkpoint(trial_model_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"models/{model_path.stem}_hparam_search_{timestamp}.pt")
    torch.save(results, results_path)
    print(f"Saved results to {results_path}")

    hparam_keys = list(results[0]["hparams"].keys())
    val_losses = [result["val_loss"] for result in results]

    n_hparams = len(hparam_keys)
    n_cols = 3
    n_rows = (n_hparams + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axs = axs.flatten()

    for (i, hparam_name) in enumerate(hparam_keys):
        hparam_values = [result['hparams'][hparam_name] for result in results]

        # Scatter plot for each hyperparameter
        if hparam_name in ["lr", "weight_decay"]:
            axs[i].scatter(hparam_values, val_losses)
            axs[i].set_xscale('log')  # Set log scale for lr and weight_decay x-axis
        elif hparam_name == "loss_function":
            hparam_values = [str(hparam_value) for hparam_value in hparam_values]
            axs[i].scatter(hparam_values, val_losses)
        elif hparam_name == "activation":
            hparam_values = [str(hparam_value) for hparam_value in hparam_values]
            axs[i].scatter(hparam_values, val_losses)
        else:
            axs[i].scatter(hparam_values, val_losses)

        axs[i].set_yscale('log')  # Log scale for y-axis (validation loss)
        axs[i].set_xlabel(hparam_name)
        axs[i].set_ylabel("Validation Loss")
        axs[i].set_title(f"{hparam_name} vs Validation Loss")
        axs[i].grid(True)

    for i in range(n_hparams, len(axs)):
        fig.delaxes(axs[i])

    plt.tight_layout()

    # Save the figure with a timestamp and model path
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = Path(f"models/{model_path.stem}_hparam_search_{timestamp}.png")
    # fig_path = Path(f"mddel/hparam_search_{timestamp}.png")
    plt.savefig(fig_path)
    print(f"Saved figure to {fig_path}")

    # plt.show()



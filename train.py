from argparse import Namespace
from typing import List

import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from datasets import TripletLatent
from networks import ShallowNet
from losses import TripletLoss
from trainer import fit


def get_feature_cols(n_features: int) -> List[str]:
    feature_cols = [f"feat_{fidx}" for fidx in range(n_features)]
    return feature_cols


def main(cfg: Namespace) -> None:
    dataset_path = cfg.dataset_path
    latent_df = pd.read_csv(dataset_path)
    feature_cols = get_feature_cols(n_features=384)
    train_ds = TripletLatent(latent_df, feature_cols, True)
    kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.cuda else {}
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size)
    model = ShallowNet(384, 128)
    if cfg.cuda:
        model.cuda()
    loss_fn = TripletLoss(margin=cfg.margin)
    lr = cfg.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = cfg.n_epochs
    log_interval = 100
    fit(train_loader, train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cfg.cuda, log_interval, metrics=[], start_epoch=0)
    torch.save(model.state_dict(), "shallow_net.pt")


if __name__ == '__main__':
    opt = {
        "dataset_path": "./features.csv",
        "cuda": True,
        "n_epochs": 10,
        "lr": 1e-3,
        "batch_size": 16,
        "margin": 1
    }
    opt = Namespace(**opt)
    main(opt)

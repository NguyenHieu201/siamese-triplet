from argparse import Namespace
from typing import List
from glob import glob

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
    if cfg.train_data is not None:
        latent_train = cfg.train_data
        latent_valid = cfg.valid_data
    else:
        dataset_path = cfg.dataset_path
        latent_df = pd.read_csv(dataset_path)
    feature_cols = get_feature_cols(n_features=384)
    train_ds = TripletLatent(latent_train, feature_cols, True)
    valid_ds = TripletLatent(latent_valid, feature_cols, True)
    # kwargs = {'num_workers': 1, 'pin_memory': True} if cfg.cuda else {}
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size)
    model = ShallowNet(384, 512)
    if cfg.cuda:
        model.cuda()
    loss_fn = TripletLoss(margin=cfg.margin)
    lr = cfg.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = cfg.n_epochs
    log_interval = 100
    fit(train_loader, valid_loader, model, loss_fn, optimizer, scheduler, n_epochs, cfg.cuda, log_interval, metrics=[], start_epoch=0)
    torch.save(model.state_dict(), "latest_shallow_net.pt")


if __name__ == '__main__':
    personpath_csv = glob("./datasets/personpath/*.csv")
    train_csvs = personpath_csv[:10]
    valid_csvs = personpath_csv[10:12]
    # train
    train_dfs = []
    for csv_path in train_csvs:
        df = pd.read_csv(csv_path)
        train_dfs.append(df)
    train_dfs = pd.concat(train_dfs)
    train_dfs.name = train_dfs.name.astype("category")
    print("INFO: Loading training data successfully")
    # valid
    valid_dfs = []
    for csv_path in valid_csvs:
        df = pd.read_csv(csv_path)
        valid_dfs.append(df)
    valid_dfs = pd.concat(valid_dfs)
    valid_dfs.name = valid_dfs.name.astype("category")
    print("INFO: Loading validation data successfully")
    print(f"INFO: Train/Valid: {len(train_dfs)} / {len(valid_dfs)}")
    opt = {
        # "dataset_path": "./features.csv",
        "cuda": True,
        "n_epochs": 5,
        "lr": 1e-3,
        "batch_size": 16,
        "margin": 1,
        "train_data": train_dfs,
        "valid_data": valid_dfs
    }
    opt = Namespace(**opt)
    main(opt)

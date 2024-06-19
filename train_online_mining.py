from argparse import Namespace
from typing import List
from glob import glob

import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from datasets import TripletLatent, TabularDataset, BalancedBatchSampler
from networks import ShallowNet, ShallowEmbeddNet
from losses import TripletLoss, OnlineTripletLoss
from trainer import fit
from utils import SemihardNegativeTripletSelector
from metrics import AverageNonzeroTripletsMetric


def get_feature_cols(n_features: int) -> List[str]:
    feature_cols = [f"feat_{fidx}" for fidx in range(n_features)]
    return feature_cols


def main(cfg: Namespace) -> None:
    if cfg.train_data is not None:
        latent_train = cfg.train_data
        latent_valid = cfg.valid_data
    feature_cols = get_feature_cols(n_features=384)
    train_ds = TabularDataset(latent_train, feature_cols, train=True)
    valid_ds = TabularDataset(latent_valid, feature_cols, train=False)
    train_batch_sampler = BalancedBatchSampler(train_ds.train_labels, n_classes=16, n_samples=32)
    valid_batch_sampler = BalancedBatchSampler(valid_ds.train_labels, n_classes=16, n_samples=32)
    online_train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler)
    online_valid_loader = DataLoader(valid_ds, batch_sampler=valid_batch_sampler)
    
    margin = 1
    model = ShallowEmbeddNet(384, 512)
    model.cuda()
    loss_fn = OnlineTripletLoss(margin, SemihardNegativeTripletSelector(margin, cpu=False))
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    n_epochs = 20
    log_interval = 200
    fit(online_train_loader, online_valid_loader, model, loss_fn, optimizer, scheduler, 
        n_epochs, True, log_interval, metrics=[AverageNonzeroTripletsMetric()])
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

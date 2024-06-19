import os.path as osp
import itertools

import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from lapsolver import solve_dense
from tqdm import tqdm

from networks import ShallowNet


def test_spm(feat_df: pd.DataFrame):
    """_summary_

    Args:
        feat_df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    prefixes = ["201", "202", "203", "204", "205", "206", "207", "208"]
    feat_df["filename"] = feat_df["filename"].apply(lambda filepath: osp.basename(filepath).replace(".jpg", ""))
    feat_df["frame"] = feat_df["filename"].apply(lambda filename: int(filename.split("_")[1]))
    frames = feat_df.frame.unique()
    cnt_min = 0
    cnt_lap = 0
    cnt_all = 0
    embedd_feat_cols = [f"feat_{idx}" for idx in range(512)]
    for frame_idx in tqdm(frames):
        for cam1, cam2 in itertools.combinations(prefixes, r=2):
            filename1 = f"{cam1}_{frame_idx}"
            filename2 = f"{cam2}_{frame_idx}"
            feat1 = feat_df[feat_df["filename"] == filename1][embedd_feat_cols].values.astype(float) + 1e-8
            feat2 = feat_df[feat_df["filename"] == filename2][embedd_feat_cols].values.astype(float) + 1e-8
            name1 = feat_df[feat_df["filename"] == filename1]["name"].values.astype(int)
            name2 = feat_df[feat_df["filename"] == filename2]["name"].values.astype(int)
            name1 = np.expand_dims(name1, axis=-1)
            name2 = np.expand_dims(name2, axis=-1)
            dist = cdist(feat1, feat2, metric="cosine")
            id_mask = cdist(name1, name2, metric="euclidean")
            topk = dist.argsort(axis=1)
            for i in range(min(1, name2.shape[0])):
                name_pred = name2[topk[:, i]].squeeze()
                name_gt = name1.squeeze()
                cnt_min += (name_pred == name_gt).sum()
            dist = cdist(feat1, feat2, metric="euclidean")
            idx1, idx2 = solve_dense(dist)
            cnt_all += (id_mask == 0).sum()
            if len(idx1) == 0:
                continue
            cnt_lap += (name1[idx1] == name2[idx2]).sum()
    return cnt_min, cnt_lap, cnt_all


if __name__ == "__main__":
    batch_size = 1024
    df = pd.read_csv("./feature_filename.csv")
    model = ShallowNet(384, 512)
    model.load_state_dict(torch.load("./latest_shallow_net.pt"))
    model.cuda()
    feat_cols = [f"feat_{idx}" for idx in range(384)]
    feats = df[feat_cols].values
    test_data = torch.from_numpy(feats).float()
    embedd_feats =[]
    for idx in range(0, len(df), batch_size):
        inputs = test_data[idx:idx+batch_size].to("cuda:0")
        embedds = model.get_embedding(inputs).detach().cpu().numpy()
        embedd_feats.append(embedds)
    embedd_feats = np.concatenate(embedd_feats, axis=0)
    output_df = pd.DataFrame(data=embedd_feats, columns=[f"feat_{idx}" for idx in range(512)])
    output_df["name"] = df.name
    output_df["filename"] = df.filename
    output_df.to_csv("feature_filename_model.csv", index=False)
    # output_df = pd.read_csv("./feature_filename_model.csv")
    results = test_spm(output_df)
    print(results)

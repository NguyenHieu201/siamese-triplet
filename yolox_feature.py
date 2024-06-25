import json
import os.path as osp

from tqdm import tqdm
import torch
from torch import Tensor
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yolox import YOLOPAFPN, YOLOXHead, YOLOX
from yolox_roi_feature import BoxFeatureROI


def preprocess(image: cv2.Mat, input_size: tuple[int, int], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return Tensor(padded_img)


def align_bbox(bboxes: Tensor, input_size: tuple[int, int], img_size: tuple[int, int]):
    r = min(input_size[0] / img_size[0], input_size[1] / img_size[1])
    bboxes = bboxes * r
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    return bboxes


if __name__ == "__main__":
    # hyperparams
    batch_size = 4
    in_channels = [256, 512, 1024]
    num_classes = 1
    depth = 0.33
    width = 0.5
    ckpt_path = "../ByteTrack/pretrained/bytetrack_s_mot17.pth.tar"
    data_path = "../ByteTrack/res.json"
    input_size = (800, 1440)

    # Load yolox model
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)
    model.load_state_dict(torch.load(ckpt_path)["model"])
    model.eval()
    
    # ROI pooling
    # spatial_scales = [1/2, 1/4, 1/8, 1/16, 1/32]
    # output_sizes = [32, 16, 8, 4, 2]
    # map_names = ["stem", "dark2", "dark3", "dark4", "dark5"]
    
    spatial_scales = [1/8, 1/16, 1/32]
    output_sizes = [(16, 8), (8, 4), (4, 2)]
    map_names = ["dark3", "dark4", "dark5"]
    
    box_feature_roi = BoxFeatureROI(yolox=model, spatial_scales=spatial_scales, 
                                    output_sizes=output_sizes, names=map_names)
    box_feature_roi = box_feature_roi.to("cuda:0")
    box_feature_roi = box_feature_roi.eval()
    
    # Load bboxes and inferences
    dfs ={name: [] for name in map_names}
    data = json.load(open(data_path, "r"))
    images = data["images"]
    annotations = data["annotations"]
    images = [image for image in images if osp.exists(image["file_name"])]
    images = images[::12]
    for idx in tqdm(range(0, len(images), batch_size)):
        img_paths = [image["file_name"] for image in images[idx:idx+batch_size]]
        image_ids = [image["id"] for image in images[idx:idx+batch_size]]
        imgs = [cv2.imread(img_path) for img_path in img_paths]
        inputs = [preprocess(img, input_size=input_size) for img in imgs]
        inputs = torch.stack(inputs)

        bboxes, names = [], []
        for bid, image_id in enumerate(image_ids):
            annos = [anno for anno in annotations if anno["image_id"] == image_id]
            box_ids = [anno["attributes"]["name"] for anno in annos]
            if len(annos) == 0:
                continue
            bbox = [anno["bbox"] for anno in annos]
            bbox = torch.Tensor(bbox)
            bid_col = [[bid] for _ in range(len(bbox))]
            bid_col = Tensor(bid_col)
            bbox = align_bbox(bbox, input_size=input_size, img_size=imgs[0].shape[:2])
            bbox = torch.concat([bid_col, bbox], dim=1)
            bboxes.append(bbox)
            names.extend(box_ids)
        if len(bboxes) == 0:
            continue
        bboxes = torch.concat(bboxes)
        inputs = inputs.to("cuda:0")
        bboxes = bboxes.to("cuda:0")
        features = box_feature_roi(inputs, bboxes)
        bids = bboxes[:, 0]
        # features = [feature.flatten(1) for feature in features.values()]
        # features = torch.concat(features, dim=1)
        
        # write feature to file
        for name, feature in features.items():
            feature = feature.flatten(1)
            n_features = feature.shape[1]
            sub_df = pd.DataFrame(data=feature.detach().cpu().numpy(),
                                  columns=[f"feat_{fid}" for fid in range(n_features)])
            sub_df["pos_in_batch"] = bboxes[:, 0].detach().cpu().numpy()
            sub_df["name"] = names
            sub_df["filename"] = sub_df["pos_in_batch"].apply(lambda pos_in_batch: img_paths[int(pos_in_batch)])
            dfs[name].append(sub_df)
    for name, df in dfs.items():
        df = pd.concat(df)
        df.to_csv(f"./datasets/spm_175930_dark3_168/feature_{name}.csv", index=False)
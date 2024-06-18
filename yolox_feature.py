import json
import os.path as osp

import torch
from torch import Tensor
import cv2
import numpy as np
from torchvision.ops import roi_pool
import pandas as pd

from yolox import YOLOPAFPN, YOLOXHead, YOLOX
from yolox.darknet import CSPDarknet


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
    return bboxes


def forward_fpn_feature(inputs: Tensor, backbone: CSPDarknet):
    outputs = {}
    x = backbone.stem(inputs)
    outputs["stem"] = x
    x = backbone.dark2(x)
    outputs["dark2"] = x
    x = backbone.dark3(x)
    outputs["dark3"] = x
    x = backbone.dark4(x)
    outputs["dark4"] = x
    x = backbone.dark5(x)
    outputs["dark5"] = x
    return outputs


@torch.no_grad()
def forward_feature_roi(feature_map: Tensor, input_size: tuple[int, int], bboxes: Tensor) -> Tensor:
    ratio = feature_map.shape[-2] / input_size[0]
    bboxes_map = bboxes * ratio
    bboxes_map[:, :, 2] += bboxes_map[:, :, 0]
    bboxes_map[:, :, 3] += bboxes_map[:, :, 1]
    bboxes_map = [bboxes_map.squeeze(0)]
    roi_features: Tensor = roi_pool(input=feature_map, boxes=bboxes_map, output_size=(2, 2))
    return roi_features


def get_features(inputs: Tensor, model: YOLOX, bboxes: Tensor,
                 input_size: tuple[int, int], image_size: tuple[int, int]):
    feature_maps = forward_fpn_feature(inputs, model.backbone.backbone)
    map1, map2 = feature_maps["stem"], feature_maps["dark2"]
    bboxes = align_bbox(bboxes, input_size, image_size)
    feature1 = forward_feature_roi(map1, input_size, bboxes)
    feature2 = forward_feature_roi(map2, input_size, bboxes)
    feat = [feature1.flatten(start_dim=1), feature2.flatten(start_dim=1)]
    feat = torch.concat(feat, dim=1)
    return feat


if __name__ == "__main__":
    in_channels = [256, 512, 1024]
    num_classes = 1
    depth = 0.33
    width = 0.5
    ckpt_path = "../ByteTrack/pretrained/bytetrack_s_mot17.pth.tar"
    data_path = "/home/hieu/Downloads/project_spm_121044_with_identity-2024_01_16_09_42_12-coco 1.0/annotations/instances_default.json"
    input_size = (800, 1440)

    backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
    head = YOLOXHead(num_classes, width, in_channels=in_channels)
    model = YOLOX(backbone, head)
    model.load_state_dict(torch.load(ckpt_path)["model"])
    model.eval()
    
    
    results = []
    data = json.load(open(data_path, "r"))
    images = data["images"]
    annotations = data["annotations"]
    for image in images:
        # img_path = image["file_name"]
        img_path = osp.join("/home/hieu/Downloads/project_spm_121044_with_identity-2024_01_16_09_42_12-coco 1.0/images",
                            image["file_name"])
        img = cv2.imread(img_path)
        inputs = preprocess(img, input_size=input_size)
        inputs = inputs.unsqueeze(0)
        annos = [anno for anno in annotations if anno["image_id"] == image["id"]]
        if len(annos) == 0:
            continue
        bboxes, names, filenames = [], [], []
        for anno in annos:
            bbox = anno["bbox"]
            name = anno["attributes"]["name"]
            bboxes.append(bbox)
            names.append(name)
            filenames.append(img_path)
        bboxes = [bboxes]
        bboxes = Tensor(bboxes)
        feats =get_features(inputs, model, bboxes, input_size, img.shape[:2])
        for idx, name in enumerate(names):
            feat = feats[idx]
            record = {f"feat_{fdx}": feat[fdx].item() for fdx in range(feat.shape[0])}
            record["name"] = name
            record["filename"] = img_path
            results.append(record)
    df = pd.DataFrame(results)
    df.to_csv("features_fake.csv", index=False)
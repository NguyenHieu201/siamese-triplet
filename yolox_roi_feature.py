import torch.nn as nn
from torch import Tensor
from torchvision.ops import RoIPool
import torch

from yolox import YOLOX


class FeatureMapROI(nn.Module):
    def __init__(self, spatial_scales: list[float], output_sizes: list[int], names: list[str]):
        super(FeatureMapROI, self).__init__()
        self.spatial_scales = spatial_scales
        self.roi_pools = self.__init_roi_pool(spatial_scales, output_sizes, names)
        
    def __init_roi_pool(self, spatial_scales, output_sizes, names) -> dict[str, nn.Module]:
        roi_pools = {}
        for spatial_scale, output_size, name in zip(spatial_scales, output_sizes, names):
            roi_pools[name] = RoIPool(output_size=output_size, spatial_scale=spatial_scale)
        return roi_pools
    
    def forward(self, feature_maps: dict[str, Tensor], bboxes: list[Tensor]) -> dict[str, Tensor]:
        roi_features = {}
        for name in self.roi_pools.keys():
            feature_map = feature_maps[name]
            roi_pool = self.roi_pools[name]
            roi_feature = roi_pool(feature_map, bboxes)
            roi_features[name] = roi_feature
        return roi_features
    
class YOLOXFeatureMap(nn.Module):
    def __init__(self, yolox: YOLOX):
        super(YOLOXFeatureMap, self).__init__()
        self.backbone = yolox.backbone.backbone
        
    def forward(self, inputs):
        outputs = {}
        x = self.backbone.stem(inputs)
        outputs["stem"] = x
        x = self.backbone.dark2(x)
        outputs["dark2"] = x
        x = self.backbone.dark3(x)
        outputs["dark3"] = x
        x = self.backbone.dark4(x)
        outputs["dark4"] = x
        x = self.backbone.dark5(x)
        outputs["dark5"] = x
        return outputs
    
class BoxFeatureROI(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(BoxFeatureROI, self).__init__()
        self.yolox_feature_map = YOLOXFeatureMap(kwargs["yolox"])
        self.roi_feature_map = FeatureMapROI(kwargs["spatial_scales"], kwargs["output_sizes"], kwargs["names"])
        self.keys = kwargs["names"]
        
    def forward(self, inputs, bboxes):
        with torch.no_grad():
            feature_maps = self.yolox_feature_map(inputs)
            roi_features = self.roi_feature_map(feature_maps, bboxes)
            return roi_features
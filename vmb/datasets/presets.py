import torch
from torchvision.transforms import transforms
import pytorchvideo.transforms as vtransforms

from vmb.datasets.transform import ConvertBHWCtoBCHW, ConvertBCHWtoCBHW


"""
If the Pretrained model is from 
https://pytorch.org/vision/stable/models.html#video-classification 
the following presets can be used:
"""


class VideoClassificationPresetTrain:
    def __init__(
        self,
        min_max_scale_size,
        crop_size,
        mean=(0.43216, 0.394666, 0.37645),
        std=(0.22803, 0.22145, 0.216989),
        hflip_prob=0.5,
    ):
        trans = [
            ConvertBHWCtoBCHW(),
            transforms.ConvertImageDtype(torch.float32),
            vtransforms.RandomShortSideScale(*min_max_scale_size),
        ]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend([transforms.Normalize(mean=mean, std=std),
                      transforms.RandomCrop(crop_size),
                      ConvertBCHWtoCBHW(),
                      ])
        self.transforms = transforms.Compose(trans)

    def __call__(self, x):
        return self.transforms(x)


class VideoClassificationPresetEval:
    def __init__(self,
                 min_short_side_scale,
                 crop_size,
                 mean=(0.43216, 0.394666, 0.37645),
                 std=(0.22803, 0.22145, 0.216989)):
        transform_list = [
                ConvertBHWCtoBCHW(),
                transforms.ConvertImageDtype(torch.float32),
                vtransforms.ShortSideScale(min_short_side_scale),
                transforms.Normalize(mean=mean, std=std),
                transforms.CenterCrop(crop_size),
                ConvertBCHWtoCBHW(),
            ]
        self.transforms = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transforms(x)

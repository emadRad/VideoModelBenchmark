"""Model construction functions."""

import torch
import torch.nn as nn


class ModelWrapper(nn.Module):
    """
    Wrapper class for models to change
    the number of classes in classification layer.
    """
    def __init__(self, cfg):
        super(ModelWrapper, self).__init__()
        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.model = torch.hub.load('facebookresearch/pytorchvideo',
                                    cfg.MODEL.ARCH,
                                    pretrained=cfg.MODEL.PRETRAINED)

        if cfg.MODEL.ARCH == 'mvit_base_16x4':
            num_features = self.model.head.proj.in_features
            self.model.head.proj = nn.Linear(num_features, self.num_classes, bias=True)
        elif 'x3d' in cfg.MODEL.ARCH:
            num_features = self.model.blocks[5].proj.in_features
            self.model.blocks[5].proj = nn.Linear(num_features, self.num_classes, bias=True)
        else:
            raise NotImplementedError(f'Model {cfg.MODEL.ARCH} architecture not supported.')

    def forward(self, x):
        return self.model(x)


def build_model(cfg):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in vml/config/defaults.py.
    """
    # Construct the model
    available_models = torch.hub.list('facebookresearch/pytorchvideo')
    assert cfg.MODEL.ARCH in available_models, "Model not found in facebookresearch/pytorchvideo"
    model = ModelWrapper(cfg)
    return model

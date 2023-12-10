import torch
import torch.nn as nn
from typing import List

from . import _utils_3D as utils_3D

class EncoderMixin_3D:
    """
    Add encoder functionality such as:
        - output channels specification of feature tensors (produced by encoder)
        - patching first convolution for arbitrary input channels
    """

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]

    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils_3D.patch_first_conv3d(model=self, new_in_channels=in_channels, pretrained=pretrained)

    def get_stages(self):
        """Method should be overridden in encoder"""
        raise NotImplementedError
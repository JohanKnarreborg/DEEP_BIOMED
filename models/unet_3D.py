from typing import Optional, Union, List
from models.unet_3D_utils.unet_3D.decoder import UnetDecoder_3D
from models.unet_3D_utils.encoders import get_encoder
from models.unet_3D_utils.base import SegmentationModel
from models.unet_3D_utils.base import SegmentationHead_3D, ClassificationHead


class pretrained_unet_3D(SegmentationModel):
    """
    UNET-3D model taken from: https://github.com/PUTvision/segmentation_models.pytorch.3d/tree/master
    All related utility functions, placed in models/unet_3D_utils/, are also copied from here.
    """

    def __init__(
        self,
        encoder_name: str = "resnet18_3D",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        temporal_size: int = 1,
    ): 
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder_3D(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead_3D(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
            temporal_size=temporal_size
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

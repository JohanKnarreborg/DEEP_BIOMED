import numpy as np
from skimage.measure import label as skimage_label, regionprops
from models.unetr_monai import UNETR
from models.swinunetr_monai import SwinUNETR
from models.unet_3D import pretrained_unet_3D

from monai.networks.nets import UNet

from monai.losses import MaskedDiceLoss
from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    FgBgToIndicesd,
    LabelToMaskd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandAxisFlipd,
)

def extract_label_patches(image, label, patch_size):
    """
    Extract patches from image where label is non-zero.

    For each connected component in label, extract the bounding box.
    Split the bounding box into overlapping patches of size patch_size.
    Extract the patches from image and label.
    Return the patches as a list of { 'image': ..., 'label': ... } dicts.

    Args:
        image (np.ndarray): Image to extract patches from.
        label (np.ndarray): Label to extract patches from.
        patch_size (tuple): Size of the patches to extract.

    Returns:
        list: List of patches as { 'image': ..., 'label': ..., 'mask': ... } dicts.
    """
    props = regionprops(skimage_label(label > 0))  # Extract connected components of labeled voxels
    patches = []
    for pp in props:
        # Extract bounding box for connected component
        cc_min = pp.bbox[:3]
        cc_max = pp.bbox[3:]

        # Extract patches covering the bounding box
        for z in range(cc_min[0] - patch_size[0] // 2, cc_max[0] + patch_size[0] // 2, patch_size[0] // 2):
            for y in range(cc_min[1] - patch_size[1] // 2, cc_max[1] + patch_size[1] // 2, patch_size[1] // 2):
                for x in range(cc_min[2] - patch_size[2] // 2, cc_max[2] + patch_size[2] // 2, patch_size[2] // 2):
                    # Ensure patch is within image bounds
                    z_begin = max(z, 0)
                    y_begin = max(y, 0)
                    x_begin = max(x, 0)
                    z_end = min(z + patch_size[0], image.shape[0])
                    y_end = min(y + patch_size[1], image.shape[1])
                    x_end = min(x + patch_size[2], image.shape[2])

                    patch_label = label[z_begin:z_end, y_begin:y_end, x_begin:x_end]
                    if not patch_label.any():
                        # Skip empty patches
                        continue
                    patch_image = image[z_begin:z_end, y_begin:y_end, x_begin:x_end]

                    if patch_image.shape != patch_size:
                        # Pad patch if it is smaller than patch_size
                        pad_size = [(0, 0)] * 3
                        for d in range(3):
                            pad_size[d] = (0, patch_size[d] - patch_image.shape[d])
                        patch_image = np.pad(patch_image, pad_size, 'constant', constant_values=0)
                        pad_size = [(0, 0)] * 3
                        for d in range(3):
                            pad_size[d] = (0, patch_size[d] - patch_label.shape[d])
                        patch_label = np.pad(patch_label, pad_size, 'constant', constant_values=0)

                    patches.append({ 'image': patch_image, 'label': patch_label, 'mask': patch_label > 0 })

    return patches

def get_transforms(patch_size, prob_foreground_center):
    """
    Generates transformation pipelines for training and validation datasets.

    For training, this function creates a series of MONAI transforms that include
    random cropping, rotation, and flipping, tailored to handle 3D medical images
    and their associated labels/masks. For validation, it ensures the channel
    dimension is set correctly for the images, labels, and masks.

    Args:
        - patch_size (tuple): Size of the patches to extract.
        - prob_foreground_center (float): Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)

    Returns:
        - train_transforms (Tuple[monai.transforms.Compose]): A series of MONAI transforms for training and validation.
    """
    train_transforms = Compose([
        EnsureChannelFirstd(keys=['image', 'label'], channel_dim='no_channel'),
        CopyItemsd(keys=['label'], times=1, names=['mask']),                                                  # Copy label to new image mask
        LabelToMaskd(keys=['mask'], select_labels=[1, 2], merge_channels=True),                               # Convert mask to binary mask showing where labels are
        FgBgToIndicesd(keys=['mask'], fg_postfix='_fg_indices', bg_postfix='_bg_indices'),                    # Precompute indices of labelled voxels
        RandCropByPosNegLabeld(keys=['image', 'label', 'mask'], label_key='label', spatial_size=patch_size,   # Extract random crop
                                pos=prob_foreground_center, neg=(1.0 - prob_foreground_center),
                                num_samples=1, fg_indices_key='mask_fg_indices', bg_indices_key='mask_bg_indices'),
        RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(0, 1)),                        # Randomly rotate
        RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(1, 2)),                        # Randomly rotate
        RandRotate90d(keys=['image', 'label', 'mask'], prob=0.5, spatial_axes=(0, 2)),                        # Randomly rotate
        RandAxisFlipd(keys=['image', 'label', 'mask'], prob=0.1),                                             # Randomly flip
    ])
    val_transforms = Compose([
        EnsureChannelFirstd(keys=['image', 'label', 'mask'], channel_dim='no_channel'),
    ])

    return train_transforms, val_transforms

def get_loss_fn(loss_fn_name):
    """
    Retrieves a specified loss function for model training.

    Args:
        - loss_fn_name (str): The name of the loss function to retrieve.

    Supported Loss Functions:
        - "maskedDiceLoss": Returns the MaskedDiceLoss function with background inclusion.

    Returns:
        - loss_fn (monai.losses): The loss function to use for model training.
    """
    loss_fn = {
        "maskedDiceLoss": MaskedDiceLoss(include_background=True)
    }
    return loss_fn[loss_fn_name]

def get_model(model_name):
    """
    Retrieves a specified model for model training.

    Args:
        - model_name (str): The name of the model to retrieve.

    Supported Models:
        - "unetr": Returns the UNETR model.
        - "swin_unetr": Returns the swin-unetr model.
    
    Returns:
        - model (torch.nn.Module): The model to use for model training.
    """
    model = {
        "unet": UNet,
        "pretrained_unet": pretrained_unet_3D,
        "unetr": UNETR,
        "swin_unetr": SwinUNETR
    }
    return model[model_name]

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
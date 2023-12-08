import torch
import hydra
import numpy as np
import os
from omegaconf import OmegaConf
from monai.data import CacheDataset, DataLoader

from dataset import RepeatedCacheDataset
import config_schema
from utils import extract_label_patches, get_transforms, get_loss_fn, get_model
from trainer import train_loop

torch.cuda.empty_cache()

@hydra.main(version_base=None, config_path="config")
def main(config: config_schema.ConfigSchema):
    """
    Main function for initializing and running the training of a specified deep learning model on the specified dataset.

    The function uses a Hydra-based configuration approach for setting up the training environment.
    It loads training and validation data, sets up data transformations, and initializes the model,
    optimizer, and loss function. The training process is then initiated by calling `train_loop`.

    Args:
        - config (config_schema.ConfigSchema): A configuration object, defined in 'config_schema.py'

    Returns:
        - A string indicating the path to the saved best model.
    """
    print("training with config: \n\n%s", OmegaConf.to_yaml(config))

    if "full" in config.data.data_path:
        image = torch.from_numpy(np.load(os.path.join(config.data.data_path, 'train', 'data_0.npy'))).float()
        train_label = torch.from_numpy(np.load(os.path.join(config.data.data_path, 'train', 'mask_0.npy')))
        val_label = torch.from_numpy(np.load(os.path.join(config.data.data_path, 'val', 'mask_0.npy')))
    else: # we use the cropped data
        image = torch.from_numpy(np.load(os.path.join(config.data.data_path, 'train', 'image_256_512.npy'))).float()
        train_label = torch.from_numpy(np.load(os.path.join(config.data.data_path, 'train', 'train_label_256_512.npy')))
        val_label = torch.from_numpy(np.load(os.path.join(config.data.data_path, 'val', 'val_label_256_512.npy')))

    if config.data.crop_volume_size != 0:
        new_size = config.data.crop_volume_size
        image = image[new_size:new_size*2, new_size:new_size*2, new_size:new_size*2]
        train_label = train_label[new_size:new_size*2, new_size:new_size*2, new_size:new_size*2]
        val_label = val_label[new_size:new_size*2, new_size:new_size*2, new_size:new_size*2]
    
    input_img_size = config.data.input_img_size
    patch_size = (input_img_size,) * 3
    prob_foreground_center = config.data.prob_foreground_center

    train_transforms, val_transforms = get_transforms(patch_size, prob_foreground_center)

    train_dataset = RepeatedCacheDataset(
        data=[{ 'image': image, 'label': train_label }],
        num_repeats=config.training.batches_per_epoch * config.training.train_batch_size,
        transform=train_transforms,
        num_workers=1,
        cache_rate=1.0,
        copy_cache=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.val_batch_size,
        shuffle=False,  # Don't shuffle since we use random crops
        num_workers=0
    )

    val_dataset = CacheDataset(
        data=extract_label_patches(image, val_label, patch_size),
        transform=val_transforms,
        num_workers=1,
        cache_rate=1.0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=0
    )

    loss_fn = get_loss_fn(config.training.loss_fn_name)
    model = get_model(config.model.model_type)
    if config.model.model_type == "unet":
        model = model(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.2,  
        )
    else:
        model = model(
            img_size=(input_img_size, input_img_size, input_img_size),
            spatial_dims=3,
            in_channels=1,
            out_channels=2
        )
    model.to(config.training.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)

    print('Starting training')
    trained_model_path = train_loop(config, model, train_loader, val_loader, loss_fn, optimizer)
    print('Finished training')

    return f"Best model saved at {trained_model_path}"


if __name__ == "__main__":
    config_schema.register_config()
    main()
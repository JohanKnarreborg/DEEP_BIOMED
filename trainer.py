import wandb
from time import perf_counter as time
from tqdm import tqdm
from monai.networks.utils import one_hot
from utils import count_parameters
import os
import torch
import datetime
import numpy as np
from omegaconf import OmegaConf

def train_loop(config, model, train_loader, val_loader, loss_fn, optimizer):
    """
    Executes the training loop for a given model using specified training and validation data loaders.

    Initializes a Weights & Biases (wandb) project for tracking and logging the training process.
    Handles the training and validation of the model for a given number of epochs, specified in the config.
    The function logs various metrics to wandb and saves the best model based on validation loss.

    Args:
        - config (OmegaConf): A configuration object containing settings for training, model, and data paths.
        - model (torch.nn.Module): The neural network model to be trained.
        - train_loader (DataLoader): The DataLoader for the training dataset.
        - val_loader (DataLoader): The DataLoader for the validation dataset.
        - loss_fn (callable): The loss function used for model training.
        - optimizer (torch.optim.Optimizer): The optimizer used for model training.

    Returns:
        - model_save_path (str): The path to the saved best model based on validation loss.
    """
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    temp_config = OmegaConf.to_container(config, resolve=True)
    run = wandb.init(project="deep_med_epfl", 
                    entity="deep_med_epfl", 
                    name=str(config.run_name + current_datetime), 
                    config=temp_config,
                    notes=f"crop_volume: {config.data.crop_volume_size}, input_patches_size: {config.data.input_img_size}")

    # creating a folder for the model based on current date and time (dd-mm-yyyy_hh-mm)
    model_save_folder = os.path.join(config.data.data_path.rsplit('/', 1)[0], config.run_name, current_datetime)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    else:
        raise ValueError('Model folder already exists - exiting')
    
    # log number of model parameters
    total_params, trainable_params = count_parameters(model)
    run.log({"total_model_params": total_params, "trainable_model_params": trainable_params})

    best_val_loss = float('inf')
    for epoch in range(config.training.num_epochs):
        mean_train_loss = 0
        num_samples = 0
        step = 0
        t0 = time()
        model.train()
        for batch in tqdm(train_loader):
            if config.model.pretrained_model is not None:
                image_b = torch.repeat_interleave(batch["image"], 3, dim=1).to(config.training.device)
            else:
                image_b = batch['image'].as_tensor().to(config.training.device)#, non_blocking=True
            label = batch['label'].as_tensor().to(config.training.device)
            mask = batch['mask'].as_tensor().to(config.training.device)

            label = one_hot(label, num_classes=3)
            label = label[:, 1:]

            pred = model(image_b)
            loss = loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=None)

            mean_train_loss += loss.detach() * len(image_b)
            run.log({"batch_train_loss": loss.item()})
            num_samples += len(image_b)
            step += 1

        train_time = time() - t0
        mean_train_loss = mean_train_loss / num_samples
        run.log(
            {"epoch": epoch,
            "epoch_train_loss": mean_train_loss.item(),
            "epoch_train_time": train_time}
        )

        if epoch % 10 == 0:
            mean_val_loss = 0
            num_samples = 0
            step = 0
            t0 = time()
            model.eval()
            for batch in tqdm(val_loader):
                if config.model.pretrained_model is not None:
                    image_b = torch.repeat_interleave(batch["image"], 3, dim=1).to(config.training.device)
                else:
                    image_b = batch['image'].as_tensor().to(config.training.device)#, non_blocking=True
                label = batch['label'].as_tensor().to(config.training.device)
                mask = batch['mask'].as_tensor().to(config.training.device)

                with torch.no_grad():
                    label = one_hot(label, num_classes=3)
                    label = label[:, 1:]

                    pred = model(image_b)
                    loss = loss_fn(input=pred.softmax(dim=1), target=label, mask=mask)

                mean_val_loss += loss * len(image_b)
                num_samples += len(image_b)
                step += 1
            
            # log image_b, its label and its prediction
            # print(f"{image_b.shape=}")
            # print(f"{label.shape=}")
            # print(f"{pred.shape=}")
            # logging_image = image_b[-1].permute(1, 2, 3, 0).detach().cpu().numpy() # from [16, 1, 32, 32, 32] to [32, 32, 32, 1]
            # logging_label = label[-1].permute(1, 2, 3, 0).detach().cpu().numpy() # from [16, 2, 32, 32, 32] to [32, 32, 32, 2]
            # logging_pred = pred[-1].permute(1, 2, 3, 0).detach().cpu().numpy() # from [16, 2, 32, 32, 32] to [32, 32, 32, 2]
            # print(f"{logging_image.shape=}")
            # print(f"{logging_label.shape=}")
            # print(f"{logging_pred.shape=}")
            # run.log(
            #     {"epoch_val_loss": mean_val_loss, 
            #     "image": wandb.Object3D(logging_image), 
            #     "label": wandb.Object3D(logging_label), 
            #     "prediction": wandb.Object3D(logging_pred)} 
            # )

            val_time = time() - t0
            mean_val_loss = mean_val_loss / num_samples
            run.log({"epoch_val_loss": mean_val_loss.item()})

            # save best model
            if mean_val_loss.item() < best_val_loss:
                print('New best model checkpoint, epoch', epoch, 'val loss', mean_val_loss.item())
                best_val_loss = mean_val_loss
                model_save_path = os.path.join(model_save_folder, f'best_model_epoch{epoch}.pt')
                model_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_val_loss.item(),
                    }
                
                
        print('Epoch', epoch + 1, 'train loss', mean_train_loss.item(), 'val loss', mean_val_loss.item(), 'train time', train_time, 'seconds val time', val_time, 'seconds')
    # save the best model in the end
    torch.save(model_dict, model_save_path)
    return model_save_path
import wandb
from time import perf_counter as time
from tqdm import tqdm
from monai.networks.utils import one_hot
import os
import torch
import datetime

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
    # Function implementation...

    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    print(config)
    wandb.init(project="deep_med_epfl", name=str(config.run_name + current_datetime), config=config)
    wandb.log(
        {"device": config.training.device,
        "model_type": config.model.model_type}
    )

    # creating a folder for the model based on current date and time (dd-mm-yyyy_hh-mm)
    model_save_folder = os.path.join(config.data.data_path.rsplit('/', 1)[0], config.run_name, current_datetime)
    if not os.path.exists(model_save_folder):
        os.makedirs(model_save_folder)
    else:
        raise ValueError('Model folder already exists - exiting')

    best_val_loss = float('inf')
    for epoch in range(config.training.num_epochs):
        mean_train_loss = 0
        num_samples = 0
        step = 0
        t0 = time()
        model.train()
        for batch in tqdm(train_loader):
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
            wandb.log({"batch_train_loss": loss})
            num_samples += len(image_b)
            step += 1

        train_time = time() - t0
        mean_train_loss = mean_train_loss / num_samples
        wandb.log(
            {"epoch": epoch,
            "epoch_train_loss": mean_train_loss,
            "epoch_train_time": train_time}
        )

        if epoch % 10 == 0:
            mean_val_loss = 0
            num_samples = 0
            step = 0
            t0 = time()
            model.eval()
            for batch in tqdm(val_loader):
                image_b = batch['image'].as_tensor().to(config.training.device)
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
            wandb.log(
                {"epoch_val_loss": mean_val_loss, 
                "image": wandb.Image(image_b), 
                "label": wandb.Image(label), 
                "prediction": wandb.Image(pred)}
            )

            val_time = time() - t0
            mean_val_loss = mean_val_loss / num_samples

            # save best model
            if mean_val_loss.item() < best_val_loss:
                print('Saving best model checkpoint, epoch', epoch, 'val loss', mean_val_loss.item())
                best_val_loss = mean_val_loss
                model_save_path = os.path.join(model_save_folder, f'best_model_epoch{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': mean_val_loss,
                    }, model_save_path)
                
        print('Epoch', epoch + 1, 'train loss', mean_train_loss.item(), 'val loss', mean_val_loss.item(), 'train time', train_time, 'seconds val time', val_time, 'seconds')

    return model_save_path
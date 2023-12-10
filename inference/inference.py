import argparse
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
import numpy as np
import os
import sys
import torch
from skimage.io import imsave

# Construct the absolute path to the 'models' directory
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))
# Add the 'models' directory to the Python path
sys.path.append(models_dir)

from models.unetr_monai import UNETR
from models.unet_3D import pretrained_unet_3D

def main(model_type, data_path, wandb_runtime):
    input_img_size = 64
    PATCH_SIZE = (input_img_size,) * 3
    INFERENCE_BATCH_SIZE = 16
    WINDOW_OVERLAP = 0.5

    data_path = data_path # './covid_data.nosync/full_data/train/data_0.npy'
    print("Start loading of data")
    image = torch.from_numpy(np.load(data_path)).float()
    print("Finished loading of data")
    #image = image[0:64, 0:64, 0:64]

    print("Loading model init")
    if model_type == "unet":
        model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            dropout=0.2,  
        )
    elif model_type == "pretrained_unet" or model_type == "pretrained_unet_freeze":
        model = pretrained_unet_3D(
            encoder_name = "resnet18_3D",
            encoder_weights = "imagenet",
            classes = 2,
            activation = "sigmoid",
            in_channels = 1
        )
    else:
        model = UNETR(
            img_size=(input_img_size, input_img_size, input_img_size),
            spatial_dims=3,
            in_channels=1,
            out_channels=2
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Setting device to: {device}")

    path_to_model = f"./covid_data.nosync/{model_type}/{wandb_runtime}/"
    # list all files in directory
    files = os.listdir(path_to_model)
    print(f"Loading model checkpoint at {os.path.join(path_to_model, files[0])}")
    checkpoint = torch.load(os.path.join(path_to_model, files[0]), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Starting inference")
    model.eval()
    model.to(device)
    with torch.no_grad():
        # Evaluate the model on the image using MONAI sliding window inference
        pred = sliding_window_inference(
            image[None, None],
            PATCH_SIZE,
            INFERENCE_BATCH_SIZE,
            lambda x: model(torch.repeat_interleave(x, 3, dim=1).to(device) if model_type == "pretrained_unet" or model_type == "pretrained_unet_freeze" else x.to(device)).softmax(dim=1).cpu(),  # send patch to GPU, run model, call softmax, send result back to CPU
            overlap=WINDOW_OVERLAP,
            mode='gaussian',
            progress=True,
        )
    pred = pred.numpy()
    print("Inference done!")

    #Convert to 0-255 and SAVING
    pred = np.uint8(pred[0, 0] * 255)
    imsave(f'inference/inference_output/prediction_' + {model_type} + {wandb_runtime} + '.tiff', pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="unetr", help="Type of model to use for inference")
    parser.add_argument('--data_path', type=str, default="./covid_data.nosync/crop_data/train/image_256_512.npy", help="Path to data")
    parser.add_argument('--wandb_runtime', type=str, default="2023-12-10_12-22", help="Wandb runtime")

    args = parser.parse_args()
    main(args.model_type, args.data_path, args.wandb_runtime)

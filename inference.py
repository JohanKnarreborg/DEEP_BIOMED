from monai.networks.nets import UNETR
import os
import numpy as np
import torch
from skimage.io import imsave

def main():
    model_name = 'best_model_img_size_64_epoch170.pt'

    input_img_size = 64
    data_path = './covid_data.nosync/crop_data/train/image_256_512.npy'
    image = torch.from_numpy(np.load(data_path)).float()
    #image = image[0:64, 0:64, 0:64]

    PATCH_SIZE=(input_img_size,) * 3         # Size of crops
    PROB_FOREGROUND_CENTER=0.95 # Probability that center of crop is a labeled foreground voxel (ensures the crops often contain a label)

    model = UNETR(
        img_size=(input_img_size, input_img_size, input_img_size),
        spatial_dims=3,
        in_channels=1,
        out_channels=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    checkpoint = torch.load('trained_models/'+model_name,map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['model_state_dict'])

    from monai.inferers import sliding_window_inference

    INFERENCE_BATCH_SIZE = 16
    WINDOW_OVERLAP = 0.5

    model.eval()
    with torch.no_grad():
        # Evaluate the model on the image using MONAI sliding window inference
        pred = sliding_window_inference(
            image[None, None],
            PATCH_SIZE,
            INFERENCE_BATCH_SIZE,
            lambda x: model(x).softmax(dim=1),  # send patch to GPU, run model, call softmax, send result back to CPU
            overlap=WINDOW_OVERLAP,
            mode='gaussian',
            progress=True,
        )
    pred = pred.numpy()

    #Convert to 0-255 and SAVING

    pred = np.uint8(pred[0, 0] * 255)
    #imsave('prediction_'+model_name[:-3]+'_size512.tiff', pred)
    print(pred.shape)
    imsave('./inference_output/prediction_'+model_name[:-3]+'.tiff', pred)


if __name__ == "__main__":
    main()
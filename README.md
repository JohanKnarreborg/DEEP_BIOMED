# Deep Learning in Biomedicine @ EPFL Autumn 2023
This repository contains the code for the final exam project in Deep Learning in Biomedicine (CS-502) @ EPFL Autumn 2023.

## Project description
The project aims to develop a deep learning model that can segment blood vessels ...

## Code structure
The code of the project is developed in Python 3.8, and uses Hydra for configuration management. The code is structured as follows:
    
    ```
    DEEP_BIOMED
    │   README.md               # This file
    │   requirements.txt        # Required packages to run the main script
    │   config_schema.py        # The schema that the configuration files must follow
    │   main.py                 # Main script to run the code - sets up data, model and calls the trainer
    |   dataset.py              # A RepeatedCacheDataset class that is used to load and define the training data
    |   trainer.py              # The training loop, logging to Weights and Biases, and evaluation
    |   unetr_monai.py          # The UNETR model from MONAI - not developed by us
    |   utils.py                # Utility functions; transformations, loss function retrieval, etc.
    |   config                  # Folder that contains the Hydra configuration files to run experiments
    |   |   cloud_config.yaml   # The configuration to run the code on Google Cloud (Not working yet)
    |   |   dtuhpc.yaml         # The configuration to run the code on DTU HPC (remember to change the paths)
    |   |   localGustav.yaml    # The configuration to run the code on a local machine (remember to change the paths)

    ```

## How to run the code
To run the code, irrespective of local, cloud, hpc, or other, we recommend creating a new virtual environment with Python 3.8.\
From this environment, run the following command to install the required packages:

    pip install -r requirements.txt

To run the code you need to have the code locally from where you are trying to run it, and then place it in a folder with a `train` and `val` folder, like this (and remember to change the paths in the configuration file):
    
    ```
    data
    │   train
    │   |   data_0.npy
    |   |   mask_0.npy
    │   val
    │   |   data_0.npy
    |   |   mask_0.npy
    ```
    
Then, you can run the code with the following command:

    python main.py --config-name {name of config file (without file extension)}

This command will start the experiment and you should be able to monitor the training process on Weights and Biases.\

# Related ressources
## Project report
https://www.overleaf.com/project/656b65be3e475fad807ffd63 

## Weights and Biases project
https://wandb.ai/deep_med_epfl/deep_med_epfl

## URL to the dataset
https://drive.google.com/drive/folders/1AZvI-ITSUACy1Oik3WyULHDUDZzFydRy?usp=share_link 

## URL to the Google Cloud project
https://console.cloud.google.com/welcome?project=deeplearning-biomed

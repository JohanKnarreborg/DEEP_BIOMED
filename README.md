# Deep Learning in Biomedicine @ EPFL Autumn 2023
This repository contains the code for the final exam project in Deep Learning in Biomedicine (CS-502) @ EPFL Autumn 2023.

## Project description
The project aims to develop a deep learning model to segment the myocardium vasculature and investigate the performance difference between the traditional U-Net and the more novel U-NETR on this task. Moreover, we implement the possibility of using a pre-trained model that we report results for using fine-tuning.\
In this way, we investigate the performance of the different models to be able to reason about the effects of various model types and training strategies on a complex biomedical segmentation task.

![UNETR]([https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif](https://github.com/JohanKnarreborg/DEEP_BIOMED/blob/main/UNETR_output.gif?raw=true))

## 🚀 Quickstart 🚀
To secure reproducibility we detail in the following how to setup and run the code to achieve the results presented in the report.\
Because of the substantial amount of RAM required to load the data both training and inference have been tested in various environments to achieve the results presented in the report.\
Both for training and inference of the models, we used a high-performance computing (HPC) cluster accessible through our home university, DTU, which gave us access to running on a node with 1 NVIDIA Tesla V100 GPU.\
We have tested the code locally on M1 Macbooks (which is really slow), and we were not able to execute the code on neither Google Colab nor Google Cloud - keep this in mind if you want to run the code yourself.
### 🚂 Training 🚂
The steps of setting up the training environment and running an experiment are as follows:
1. **Clone the repository to your local machine**
2. **Create a new environment with the required packages**\
    We provide a requirements file in `training/requirements.txt`, that can be used to create a new environment with the required packages to perform a training experiment.
3. **Download data from our public Google Cloud bucket**\
    We use a data version control (dvc) setup that allows us to retrieve the data from a public Google Cloud bucket using a simple `dvc pull`-command - hence to download the full_data and crop_data folders, run this command from the root of their respective repositories.
4. **Create a new experiment on Weights and Biases**
5. **Create a new configuration file**\
    We use Hydra for configuration management, and to run an experiment you need to create a new configuration file in `training/config/` that follows the schema in `training/config_schema.py`. See `training/config/unet_config.yaml` for an example.
6. **Run the experiment**\
    To run the experiment, you need to run the following command from the root of the repository:\
    `python main.py --config-name {name of config file (without file extension)}`\
    This command will start the experiment and you should be able to monitor the training process on Weights and Biases.

### 🧠 Inference 🧠
1. **Clone the repository to your local machine**
2. **Create a new environment with the required packages**\
    We provide a requirements file in `inference/requirements.txt`, that can be used to create a new environment.
3. **Download data from our public Google Cloud bucket. Inference needs the full data.**\
    We use a data version control (dvc) setup that allows us to retrieve the data from a public Google Cloud bucket using a simple `dvc pull`-command - hence to download the full_data folder, run this command from the root of the full_data-repository.
4. **Download the model weights or train a new model**\
    If you want to run inference on a model that we have trained, you need to download the model weights. We provide these for a UNETR model trained for 200 epochs on the crop data due to computation. Download here: https://storage.googleapis.com/dlbio_storage/unetr_10122023.pt and place in a folder called `unetr` in the root of `covid_data.nosync`, i.e. `covid_data.nosync/unetr/2023-12-10_12-22/unetr.pt`, where `2023-12-10_12-22` is the `wandb_runtime` specified in the next step and is not relevant for only running inference.\
    If you want to train a new model, you need to follow the steps in the training section above.
5. **Run the inference script**\
    To run the inference script, you need to run the following command from the root of this repository:\
    `python inference/inference.py --model_type {model_type_string} --data_path {path_to_downloaded_data} --wandb_runtime {datetime_of_experiment_run}`\
    The `model_type_string` is the name of the model you want to run inference on, e.g. `unet`.\
    The `data_path` is the path to the downloaded data, e.g. `../full_data`.\
    The `wandb_runtime` is the datetime of the experiment run, e.g. `2023-12-10_12-22`.\
    Together, the model_type_string and wandb_runtime uniquely identify the experiment run on Weights and Biases, and the inference script will load the model weights.\
    The inference script will run the model on the test data and save the inference volume in the `inference_outputs` folder.
6. **Visualize the results**\
    Use a visualization tool of your choice to visualize the results. We used Tomviz, which is a 3D visualization tool for tomographic data.\

## 👨‍💻 Code structure 👩‍💻
The code of the project is developed in Python 3.8 and uses Hydra for configuration management. The code is structured as follows:
    
```
README.md               # This file
covid_data.nosync
|   crop_data           # Folder for storing 256x256x256 crop of the full volume
|   |   ...
|   full_data           # Folder for storing the full volume
|   |   ...
inference               # Folder for storing the code and outputs related to the inference step of our models
|   inference.py        # Inference script that loads a model and runs it on the test data
|   requirements2.txt   # Required packages to run the inference script
|   hpc_jobs            # Folder for storing scripts needed for running the HPC jobs
|   |   ...
|   inference_outputs   # Folder for storing the outputs of the inference step
|   |   ...
training                # Folder for storing the code and outputs related to the training of our models
|   config              # Folder that contains the Hydra configuration files to run experiments
|   |   ...
|   hpc_jobs            # Folder for storing scripts needed for running the training on HPC jobs
|   |   ...
|   models              # Folder for storing code related to initializing the models
|   |   unet_3D.py      # Pretrained 3D UNET model using a resnet backbone - not developed by us
|   |   unetr_monai.py  # UNETR model copied from Monai - not developed by us
|   |   ...
|   config_schema.py    # The schema that the configuration files must follow
|   dataset.py          # A RepeatedCacheDataset class that is used to load and define the training data
|   main.py             # Main script to run the code - sets up data, model and calls the trainer
|   trainer.py          # The training loop, logging to Weights and Biases, and evaluation
|   utils.py            # Utility functions; transformations, loss function retrieval, etc.
```

# Related resources
## Project report
https://www.overleaf.com/project/656b65be3e475fad807ffd63 

## Weights and Biases project
https://wandb.ai/deep_med_epfl/deep_med_epfl

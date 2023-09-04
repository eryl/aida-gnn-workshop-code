Code for the Graph Neural Networks AIDA Workshop
===================================================

This repository contains the practical material for the AIDA Workshop on Graph Neural Networks. The material is structured around a practical example where we extract cell locations from hispathological data and then use these graphs for prostate cancer grade assessment on the PANDa ([Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data)) dataset . The focus is on getting preparing graph data and building a model using Pytorch Geometric, not necissarily train a state-of-the-art model.

Due to time constraints, we will not perform the cell segmentation on the whole slide images during the workshop: the raw cell detection data will already be provided. The instructions for how to extract this information is given below if you want to replicate the workshop on your own data.

Learning outcomes
=================

The intended goal of this material  is to become familiar with graph neural networks from a practical point of view. The example application is classifiying histology data, but the main points should be general enough to be applicable to any application where the data are points in a Euclidean space.

After going  through the material, the student should:
 - Be able to create the necessary data structures to feed data into a neural network specified in Pytorch Geometric
 - Be able to make basic design decisions about GNNs for their application domain


Workshop material
=================

The learning material is centered around notebooks, it is a suitable platform for code-along style workshops, but for real reproducible experiments you should convert these steps to batch processing scripts.

The notebooks are designed to either 1) be run from your own computer, in which case you need to perform the installation steps below, or 2) From Google Colab in which case you need to follow the instructions "Google Colab" below.

Workshop notebooks:
 - [Graph basics](notebooks/graph_basics.ipynb) ([colab](https://colab.research.google.com/github/eryl/aida-gnn-workshop-code/blob/main/notebooks/graph_basics.ipynb))
 - [Pytorch Geometric](notebooks/pytorch_geometric.ipynb) ([colab](https://colab.research.google.com/github/eryl/aida-gnn-workshop-code/blob/main/notebooks/pytorch_geometric.ipynb))
 - [Working with data](notebooks/working_with_data.ipynb) ([colab](https://colab.research.google.com/github/eryl/aida-gnn-workshop-code/blob/main/notebooks/working_with_data.ipynb))
 - [Customizing architecture](notebooks/customizing_architectures.ipynb) ([colab](https://colab.research.google.com/github/eryl/aida-gnn-workshop-code/blob/main/notebooks/customizing_architectures.ipynb))


Installing dependencies for local use
=====================================
First clone this repository using git:

```shell
#Download the workshop material (this repository)
> git clone git@github.com:eryl/aida-gnn-workshop-code.git
```
This workshop assumes you are using [Anaconda](https://www.anaconda.com/) (or a variant like [miniforge](https://github.com/conda-forge/miniforge)), so install one if you don't have it. 

Create the environment for the workshop by running:

```shell
# Create workshop environment "aida_workshop_gnn"
> conda env create -f environment.yml
```

Which will install all necessary requirements. Start the local jupyter server by running
```
# Start the jupyter notebook server
> jupyter notebooks
```



Processing the image data (outside the workshop scope)
======================================================

This section explains how you can extract the cell graph using a framework like CellViT. This is included for completeness so you could replicate the whole process of the workshop for your own data of interest. In this case, we're using the open access data from the PANDa Kaggle competition: [Prostate cANcer graDe Assessment (PANDA) Challenge](https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data). This competition ran during 2020, but the data is still available for anyone who is interested.

To access the data, you need to perform a late registration for the competition which means you need to create a Kaggle account, this is left as an exercise.

The dataset is large (about 400GB uncompressed) so keep this in mind before attempting to recreate these steps. 

After you have downloaded and extracted the dataset, you need to perform cell segmentation to create a cell graph. The workshop material starts with cell locations in a `geojson` file and you need to use a cell detection framework which can generate such files. To prepare the example graphs in the workshop, we used [CellViT](https://github.com/eryl/CellViT/tree/multiprocessing_inference). Follow the install instructions in the repository README.md.

Move (or link) the `training_images` directory from the PANDa dataset to to the directory `[CellViT_directory]/PANDa/`.

Run the preproccessor on the dataset to extract all the patches which the neural network will be applied to. This is a
very long running process: on a Ryzen 7 5800X it took 50 hours using 14 processes. In the CellViT directory, after 
activating the CellViT environment, run the following command:

```bash
# Preprocess the dataset
python cellvit/preprocessing/patch_extraction/main_extraction.py --config PANDa/preprocessing_example.yaml --wsi_path PANDa/training_images/ --python_config_path configs/python/config.py
```

Now that the dataset has been preprocessed, we need to run the models on the result. Do this with the following command:

```bash
# Detect cells
python /home/erik/src/CellViT/scripts/cell_detection.py --model CellViT-256-x40.pth --enforce_amp --torch_compile --batch_size 16 --geojson --n_postprocess_workers 6 --n_dataloader_workers 2 process_dataset --wsi_paths PANDa/train_images --patch_dataset_path PANDa/output/preprocessing --wsi_extension tiff
```

You can adjust the batch size and number of workers to suit your computer. Postprocessing tends to be a bottleneck, so if this runs on a server with many CPU cores it's a good idea to increase it.
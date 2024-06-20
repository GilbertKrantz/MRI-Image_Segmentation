# MRI-Image_Segmentation

## General Info
This repository contains the code for the segmentation of MRI images using U-Net architecture. The dataset used for training the model is the [MRI Image Semantic Segmentation Dataset](https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation). The model is trained on the training set of the dataset and tested on the validation set. The model is trained using the [PyTorch](https://pytorch.org/) library.

## Used Models
All of the model thats used for the segmentation of MRI images is derived from U-Net Architecture from [Segmentation Model Pytorch](https://github.com/qubvel/segmentation_models.pytorch) with [imagenet] as the starting encoder weights. The models that's used are :
1. U-Net (EfficientNet-b0)
2. EfficientUNet++ (timm-efficientnet-b5)

We also do some modification on the model, which is adding some layers to the model to make it more complex and have more parameters to train.
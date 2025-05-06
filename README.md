## Overview 
This project implements an interactive image segmentation model using a modified U-Net architecture with a ResNet-34 backbone, named MFPResNetUNet. The model takes an image and synthetic user clicks (foreground and background) as input to predict a binary segmentation mask. The project leverages the LVIS dataset for training and the BSDS500 dataset for evaluation. It includes data preprocessing, model training, and evaluation with metrics like IoU and Dice scores.

##  Features

Model Architecture: A U-Net-style model with a pre-trained ResNet-34 encoder, modified to accept 5 input channels (3 for RGB image, 2 for click maps).
Interactive Segmentation: Simulates user interaction by generating synthetic foreground and background clicks.

Datasets:
LVIS: Used for training with COCO-style annotations.
BSDS500: Used for evaluation with ground truth segmentation masks.


Evaluation Metrics: IoU (Intersection over Union) and Dice coefficient to measure segmentation performance.
Visualization: Saves triplet images (input, ground truth, prediction) for visual inspection.

## Installation

Clone the Repository (if applicable):
git clone <repository-url>
cd <repository-directory>

Note: If using Kaggle, this step is not needed as the notebook is already in the environment.

Install Dependencies:The notebook includes a cell to install required packages:
!pip install albumentations pycocotools --quiet

Ensure this cell is executed before running the rest of the notebook.

##  Dataset Setup

The notebook uses two Kaggle datasets:
LVIS v1: Located at /kaggle/input/lvis-v1.
BSDS500: Located at /kaggle/input/berkeley-segmentation-dataset-500-bsds500.

Key Sections:

Data Loading: Loads LVIS for training and BSDS500 for testing.
Model Definition: Defines the MFPResNetUNet and MFPDiscriminator (though the latter is unused).
Training: Trains the model for 20 epochs with a weighted BCE loss to handle class imbalance.
Evaluation: Computes IoU and Dice scores and saves triplet images (input, ground truth, prediction).


## Datasets

LVIS v1:
Used for training.
Contains images and COCO-style annotations.
Path: /kaggle/input/lvis-v1/lvis_v1_val/.


BSDS500:
Used for evaluation.
Contains test images and ground truth segmentation masks in .mat format.
Path: /kaggle/input/berkeley-segmentation-dataset-500-bsds500/.
Ground truth masks are converted to PNG format for easier handling.



## Model Details

Architecture: MFPResNetUNet combines a ResNet-34 encoder with a U-Net decoder, using skip connections to preserve spatial details.
Input: 5 channels (RGB image + 2 click maps for foreground and background).
Output: Binary segmentation mask (1 channel, values in [0, 1] after sigmoid).
Training:
Optimizer: Adam with a learning rate of 1e-3.
Loss: Weighted BCE loss (nn.BCEWithLogitsLoss with pos_weight=5.0 to address class imbalance).
Epochs: 20, with a learning rate scheduler (StepLR, step size 5, gamma 0.1).


## Results

Evaluation on the first 5 batches of the BSDS500 test set yielded the following metrics:

Dice Score: 0.7994

IoU Score: 0.6659

Precision: 0.7917

Recall: 0.8073

Accuracy: 0.9255

Below is the confusion matrix for these batches:


![confmat_5batches 2](https://github.com/user-attachments/assets/06f89d77-f330-42b7-b586-d25149f9fafc)


## Future Improvements

Integrate the MFPDiscriminator to train with a GAN setup.
Add data augmentation (e.g., random flips, rotations) to improve model generalization.
Experiment with different click strategies (e.g., more clicks, adaptive click placement).

License
This project is for educational purposes and uses datasets available on Kaggle. Ensure compliance with the respective dataset licenses (LVIS, BSDS500) when using this code.
Acknowledgments

LVIS Dataset: Provided by the LVIS team.
BSDS500 Dataset: Provided by the Berkeley Vision Group.
Kaggle: For hosting the datasets and providing the computation environment.


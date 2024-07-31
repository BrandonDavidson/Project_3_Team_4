 # Project_3_Team_4

# See The Signs

![Road_Signs-2226x1113](https://github.com/user-attachments/assets/06135577-80e6-4926-8a82-8b6c5351c83a)

During this time we wanted to take a look at something that is used everyday and make it better. The problem that the group decided to take on is improving self-driving cars. The problem that most self-driving cars run into is that you're not suppose to use the self driving function in urban areas and are just for the highway usage. We believe that the issue that causes this is that the car has trouble iditifying all of the different variables that can happen while in high traffic areas. Some of those variables are the people entering the street from any angle, multiple signs that need to be registered and processed, other cars on the road, recognizing colors on street lights and hazards like debris. By making our model recognize these variables sooner we believe that when applied onto a car that has the self driving feature it can be used more in urban areas.

This project implements a Convolutional Neural Network (CNN) for classifying traffic signs. It includes data exploration, preprocessing, model training, and evaluation.

## Table of Contents

1. [Setup and Data Loading](#setup-and-data-loading)
2. [Data Exploration](#data-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Visualization](#visualization)
8. [Model Testing](#model-testing)

## Setup and Data Loading

The programs that were used in training this model was TensorFlow and Keras for building and training the CNN. Data is loaded from train, validation, and test directories.

## Data Exploration

Includes functions to explore the dataset, count images, and extract labels. 

## Data Preprocessing

Data is preprocessed using TensorFlow's data pipeline, including resizing images and converting labels.

## Model Architecture

The CNN architecture consists of:
- 3 convolutional blocks (Conv2D + MaxPooling2D)
- Flatten layer
- Dense layer with dropout
- Output layer with softmax activation

## Training

The model is trained using:
- Adam optimizer
- Sparse categorical crossentropy loss
- Accuracy metric

## Evaluation

The model is evaluated on a test set, and various metrics are calculated:
- Test accuracy
- Classification report
- Confusion matrix

## Visualization

Includes visualizations for:
- Confusion matrix
- ROC curve and AUC
- Learning curves
- Class-wise accuracy

## Model Testing

Provides functionality to test the model on individual images and visualize results.

## Usage

To use this project:

1. Ensure you have the required libraries installed.
2. Prepare your dataset in the specified directory structure.
3. Run the scripts in order, from data exploration to model testing.

## Requirements

- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- Scikit-learn







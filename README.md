# PyTorch CNN Generator

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-brightgreen.svg)](https://github.com/yourusername/pytorch-cnn-generator/releases)

This repository contains the code and configuration files for the PyTorch CNN Generator project. The project aims to generate images using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Model Architecture](#model-architecture)
- [Model Evaluation](#model-evaluation)
- [License](#license)

## Introduction

The PyTorch CNN Generator project addresses the challenge of generating realistic images using a Convolutional Neural Network (CNN). By leveraging a CNN, this project builds a model that can generate images from the MNIST dataset.

This project is structured into several key stages:

1. **Data Loading and Preprocessing**: Loading and preprocessing the MNIST dataset.
2. **Model Definition**: Defining the CNN model.
3. **Model Compilation**: Compiling the CNN model.
4. **Model Training**: Training the CNN model on the MNIST dataset.
5. **Model Evaluation**: Evaluating the performance of the trained CNN model.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Configuration

The configuration file `config/config.yaml` contains various settings and hyperparameters for the project. Modify this file to adjust the behavior of the CNN model.

## Usage

Run the main script:

```bash
python main.py
```

## Code Overview

- `main.py`: The main script to run the data processing and model pipeline.
- `config/paths.py`: Defines paths used in the project.
- `src/data/reader.py`: Module to read the MNIST dataset.
- `src/data/processor.py`: Module to process the MNIST dataset.
- `src/model/classes/compiler.py`: Defines and compiles the CNN model.
- `src/model/classes/trainer.py`: Trains the CNN model.
- `src/model/classes/tuner.py`: Tunes the hyperparameters of the CNN model.
- `src/utils/setup.py`: Sets up the project configuration and paths.
- `src/utils/check_gpu.py`: Utility to check GPU availability.
- `src/utils/config.py`: Utility to read the configuration file.

## Model Architecture

The CNN model architecture in this project is designed to generate images from the MNIST dataset. Here's a summary of its architecture and the purpose of each layer:

### Fully Connected Layers

- **Linear**: Fully connected layers with specified input and output features.
- **BatchNorm1d**: Normalizes the output of the previous layer to improve training stability.
- **Dropout**: Randomly drops units to prevent overfitting.

### Deconvolutional Layers

- **ConvTranspose2d**: Applies transposed convolution to upsample the feature maps.
- **BatchNorm2d**: Normalizes the output of the previous layer.
- **Dropout2d**: Randomly drops entire channels to prevent overfitting.

### Activation Function

- **Leaky ReLU**: Activation function used in the model.

### Optimizer

- **RMSprop**: Optimizes the model with a learning rate of 0.002.

## Model Evaluation

Evaluating the CNN model involves the following metrics:

1. **Mean Absolute Error (MAE)**: Measures the average absolute difference between the predicted and actual values.
2. **Mean Squared Error (MSE)**: Measures the average squared difference between the predicted and actual values.
3. **Structural Similarity Index (SSIM)**: Measures the similarity between the predicted and actual images.
4. **Peak Signal-to-Noise Ratio (PSNR)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.

## License

This project is licensed under the MIT License.
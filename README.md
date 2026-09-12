# Rice Type Classifier using PyTorch

A binary classification model built with PyTorch that predicts rice type (Cammeo vs. Osmancik) from morphological features of rice grains.

## Overview

This project trains a simple feed-forward neural network to classify rice grains based on measurements extracted from grain images, such as area, perimeter, and shape descriptors. The dataset comes from the [Rice Type Classification dataset on Kaggle](https://www.kaggle.com/datasets/mssmartypants/rice-type-classification).

## Dataset

The dataset contains geometric features extracted from images of rice grains, including:

- Area
- Major/Minor Axis Length
- Eccentricity
- Convex Area
- Equivalent Diameter
- Extent
- Perimeter
- Roundness
- Aspect Ratio

The target column, `Class`, is binary and indicates the rice variety.

## Model

A simple fully connected neural network:

- **Input layer:** 10 features → 10 hidden neurons
- **Output layer:** 10 hidden neurons → 1 output neuron
- **Activation:** Sigmoid (for binary classification)
- **Loss function:** Binary Cross-Entropy (`BCELoss`)
- **Optimizer:** Adam (learning rate = 0.001)

## Workflow

1. **Data loading** – Download the dataset via `kagglehub` and load it into a pandas DataFrame.
2. **Cleaning** – Drop missing values and remove the non-predictive `id` column.
3. **Splitting** – Split into training (70%), validation (15%), and test (15%) sets.
4. **Scaling** – Normalize features using `MaxAbsScaler`.
5. **Dataset/DataLoader** – Wrap data in a custom PyTorch `Dataset` and batch it with `DataLoader`.
6. **Training** – Train for 10 epochs, tracking loss and accuracy on both training and validation sets.
7. **Evaluation** – Measure final accuracy on the held-out test set.
8. **Visualization** – Plot training vs. validation loss and accuracy curves.
9. **Inference** – Run a prediction on a manually constructed example input.

## Requirements

```
torch
torchsummary
scikit-learn
pandas
numpy
matplotlib
kagglehub
```

Install with:

```bash
pip install torch torchsummary scikit-learn pandas numpy matplotlib kagglehub
```

## Usage

Open `Rice_type_classifier_using_Pytorch.ipynb` in Jupyter or Google Colab and run the cells in order. The notebook will:

1. Download the dataset automatically via `kagglehub`
2. Preprocess and split the data
3. Train the model
4. Report test accuracy and display training curves

## Results

The model is evaluated on a held-out test set, with accuracy printed at the end of training. See the notebook's output cells for the exact reported values and loss/accuracy plots.

## License

This project is for educational purposes.

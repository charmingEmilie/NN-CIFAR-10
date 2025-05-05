# NN-CIFAR-10

This repository implements a three-layer neural network for image classification on the CIFAR-10 dataset. The code includes data loading, model definition, training, hyperparameter search, and evaluation modules.

## Features

- Three-layer neural network with Sigmoid activation
- Batch training with learning rate decay
- L2 regularization
- Hyperparameter search
- Training progress visualization
- Weight visualization

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- scikit-learn
- CIFAR-10 dataset

## Installation

Clone this repository:

```bash
git clone https://github.com/charmingEmilie/NN-CIFAR-10.git
cd NN-CIFAR-10
```

Install required packages:

```bash
pip install numpy matplotlib scikit-learn
```

## Dataset Preparation

You can download the dataset from the [CIFAR-10 website](https://www.cs.toronto.edu/~kriz/cifar.html) and place it in the correct directory.

## Usage

### 1. Training the Model

To train the model with optimal hyperparameters:

```python
# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10('/path/to/cifar-10-batches-py')

# Initialize model with best parameters (found through hyperparameter search)
best_params = {'hs': 512, 'lr': 0.001, 'reg': 0.0001}
model = ThreeLayerNet(3072, best_params['hs'], 10, reg=best_params['reg'])

# Train the model
history = train(model, X_train, y_train, X_val, y_val,
               lr=best_params['lr'], batch_size=128, epochs=200)
```

### 2. Hyperparameter Search

To perform hyperparameter search (recommended on a subset of data first):

```python
best_params = hyperparameter_search(X_train[:10000], y_train[:10000], 
                                  X_val[:2000], y_val[:2000])
```

### 3. Testing the Model

To evaluate the trained model on the test set:

```python
test('best_model.npz', X_test, y_test)
```

### 4. Visualizing Results

The training script automatically generates and saves:

- Loss curve plot (`loss-acc-200.png`)
- Validation accuracy plot
- First layer weights visualization (`weight-200.png`)

## File Structure

```
.
├── README.md               # This file
├── cifar10_nn.py           # Main implementation file
├── best_model.npz          # Saved model weights (after training)
├── loss-acc-200.png        # Training curves
└── weight-200.png          # Weight visualization
```

## Implementation Details

### Model Architecture

- Input layer: 3072 units (32x32x3 CIFAR-10 images)
- Hidden layer: 512 units (configurable) with ReLU activation
- Output layer: 10 units (CIFAR-10 classes) with Softmax

### Model parameters

百度网盘链接: https://pan.baidu.com/s/1FNxZXyDoyE1xOi7DzamJbw?pwd=7m3n 提取码: 7m3n

### Training Process

- Batch size: 128
- Learning rate: 0.001 with decay factor 0.95 every 10 epochs
- L2 regularization coefficient: 0.0001
- Training epochs: 200


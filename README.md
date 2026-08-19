# Deep Learning Laboratory Exercises

This repository contains my solutions and experiments from the Deep Learning course at the University of Zagreb, Faculty of Electrical Engineering and Computing (FER).

The laboratory exercises cover fundamental deep learning concepts, implementation of neural network components from scratch, convolutional neural networks, recurrent neural networks, natural language processing, and metric learning using PyTorch.

## Labs Overview

### Lab 1 — Fully Connected Deep Models and PyTorch

The first laboratory focused on the fundamentals of deep learning and automatic differentiation using PyTorch.

Main topics and implementations:

- Forward and backward propagation in multilayer neural networks
- Manual implementation of a two-layer fully connected classifier using NumPy
- Analytical gradient computation and comparison with numerical/automatic gradients
- Linear regression using PyTorch autograd and gradient descent
- Multiclass logistic regression implemented as a `torch.nn.Module`
- L2 regularization and experiments with different regularization strengths
- Configurable fully connected deep models with arbitrary depth
- Parameter counting and comparison of different network architectures
- Comparison of ReLU and sigmoid activations
- Comparison between deep neural networks and RBF kernel SVM
- Classification experiments on synthetic Gaussian-mixture datasets
- MNIST classification using increasingly deep fully connected networks

**Technologies:** Python, NumPy, PyTorch, scikit-learn, Matplotlib

---

### Lab 2 — Convolutional Neural Networks

The second laboratory introduced convolutional neural networks and their implementation at both low and high levels.

Main topics and implementations:

- Implementation of fully connected, ReLU and Softmax Cross-Entropy layers
- Forward and backward propagation through CNN components
- Gradient checking
- `im2col` / `col2im` representation for efficient convolution
- L2 regularization of CNN parameters
- Comparison of a manually implemented CNN with an equivalent PyTorch model
- Visualization of convolutional filters during training
- Training and evaluation using mini-batches
- Learning-rate scheduling
- Performance analysis using:
  - accuracy
  - precision
  - recall
  - confusion matrix
  - training and validation loss
- Image classification on CIFAR-10
- Visualization of incorrectly classified examples and learned filters

**Technologies:** Python, PyTorch, torchvision, NumPy, Cython, scikit-image, Matplotlib

---

### Lab 3 — Sentiment Analysis with Recurrent Neural Networks

The third laboratory focused on natural language processing and sentiment classification using the Stanford Sentiment Treebank (SST).

Main topics and implementations:

- Loading and preprocessing text datasets using PyTorch `Dataset` and `DataLoader`
- Vocabulary construction and token-to-index conversion
- Handling special tokens such as `<PAD>` and `<UNK>`
- Preventing data leakage by constructing the vocabulary only from the training set
- Loading and using pretrained GloVe word representations
- Padding variable-length text sequences with a custom `collate_fn`
- Implementation of a mean-pooling baseline model
- Sentiment classification using recurrent neural networks
- Experiments with:
  - vanilla RNN
  - GRU
  - LSTM
- Gradient clipping
- Comparison of different recurrent architectures
- Hyperparameter experiments involving:
  - hidden size
  - number of layers
  - dropout
  - bidirectionality
  - learning rate
  - batch size
  - vocabulary size
  - minimum word frequency
  - optimizer
  - activation function
  - pretrained word representations
- Repeated experiments with different random seeds to evaluate model stability

**Technologies:** Python, PyTorch, NumPy, GloVe, Matplotlib

---

### Lab 4 — Metric Embedding

The fourth laboratory focused on metric learning and learning useful feature representations rather than directly predicting class labels.

Main topics and implementations:

- Construction of a metric-learning dataset for MNIST
- Sampling positive and negative examples for each anchor image
- Implementation of triplet loss
- CNN-based metric embedding model
- Batch normalization, ReLU activations and convolutional layers
- Global average pooling for feature extraction
- Training with the Adam optimizer
- Classification based on distances in the learned embedding space
- Comparison with classification directly in the original image space
- Saving and loading trained model parameters
- Classification of previously unseen classes
- Visualization of learned representations using PCA
- Comparison of the original image space and learned feature space

**Technologies:** Python, PyTorch, torchvision, NumPy, Matplotlib

---

## Concepts Covered

Across the four laboratories, the following deep learning concepts were explored:

- Automatic differentiation and backpropagation
- Gradient descent and optimization
- Fully connected neural networks
- Activation functions
- Regularization
- Convolutional neural networks
- Convolution and pooling
- Recurrent neural networks
- RNN, GRU and LSTM architectures
- Word embeddings
- Natural language processing
- Metric learning
- Triplet loss
- Feature embeddings
- Hyperparameter tuning
- Model evaluation and visualization
- PCA-based representation analysis

## Tools and Libraries

- Python
- PyTorch
- torchvision
- NumPy
- scikit-learn
- SciPy
- Matplotlib
- scikit-image
- Cython

## Datasets

The laboratory exercises use several standard datasets:

- **MNIST** — handwritten digit classification and metric learning
- **CIFAR-10** — image classification
- **Stanford Sentiment Treebank (SST)** — binary sentiment classification
- **GloVe** — pretrained word representations

The datasets are not included in this repository.

## Repository Structure

```text
deep-learning-labs/
│
├── lab1/
│   └── ...
│
├── lab2/
│   └── ...
│
├── lab3/
│   └── ...
│
├── lab4/
│   └── ...
│
├── README.md
└── ...
```

## Academic Context

These laboratory exercises were completed as part of the **Deep Learning** course at the University of Zagreb, Faculty of Electrical Engineering and Computing (FER).

# Artificial Neural Networks and Deep Architectures (DD2437)

This repository contains laboratory assignments for the course DD2437 - Artificial Neural Networks and Deep Architectures at KTH Royal Institute of Technology.

## Course Overview

The course explores fundamental concepts and advanced techniques in artificial neural networks and deep learning architectures. Through hands-on laboratory exercises, implementing various neural network models and applying them to real-world problems including classification, function approximation, clustering, and generative modeling.

## Laboratory Assignments

### [Lab 1a: Single-Layer Perceptrons and Delta Rule](./lab1a/)
Introduction to basic neural network concepts with single-layer perceptrons for binary classification.

**Key Concepts:**
- Perceptron learning algorithm
- Delta rule (Widrow-Hoff rule)
- Linear separability
- Learning rate and convergence
- Data subsampling effects

**Implementation:** Perceptron and Delta rule from scratch using NumPy

### [Lab 1b: Multi-Layer Perceptrons](./lab1b/)
Implementation and exploration of multi-layer perceptrons (MLPs) for classification and function approximation.

**Key Concepts:**
- Multi-layer neural networks
- Backpropagation algorithm
- Batch vs. sequential learning
- Momentum optimization
- Time series prediction
- Regularization (L2)

**Implementation:** Custom MLP with NumPy and PyTorch implementations

### [Lab 2: RBF Networks and Self-Organizing Maps](./lab2/)
Exploration of Radial Basis Function networks and Self-Organizing Maps for various learning tasks.

**Key Concepts:**
- Radial Basis Function (RBF) networks
- Gaussian basis functions
- Self-Organizing Maps (SOMs)
- Competitive learning
- Unsupervised learning
- Traveling Salesperson Problem

**Implementation:** RBF networks and SOMs from scratch, applied to clustering and TSP

### [Lab 3: Hopfield Networks](./lab3/)
Implementation and analysis of Hopfield networks for associative memory and pattern completion.

**Key Concepts:**
- Recurrent neural networks
- Associative memory
- Hebbian learning
- Energy functions and stability
- Network capacity
- Sparse patterns

**Implementation:** Complete Hopfield network with energy analysis and capacity testing

### [Lab 4: Deep Belief Networks and Restricted Boltzmann Machines](./lab4/)
Implementation of generative models using Deep Belief Networks built from Restricted Boltzmann Machines.

**Key Concepts:**
- Restricted Boltzmann Machines (RBMs)
- Deep Belief Networks (DBNs)
- Contrastive Divergence training
- Gibbs sampling
- Unsupervised pre-training
- Generative modeling

**Implementation:** RBMs and DBNs from scratch, applied to MNIST digit recognition and generation

## Course Learning Outcomes

By completing these laboratories, students gain:

1. **Theoretical Understanding:** Deep comprehension of neural network fundamentals, from basic perceptrons to complex generative models
2. **Implementation Skills:** Ability to implement neural networks from scratch using mathematical foundations
3. **Practical Application:** Experience applying different architectures to various problem domains
4. **Analysis Capabilities:** Skills in analyzing network behavior, capacity, and performance characteristics
5. **Modern Frameworks:** Experience with both custom implementations and modern deep learning frameworks (PyTorch)

## Technical Requirements

- Python 3.x
- NumPy
- Matplotlib
- PyTorch (for Lab 1b)
- Jupyter Notebook
- Additional dependencies as specified in individual lab directories

## Repository Structure

```
├── lab1a/          # Single-layer perceptrons and Delta rule
├── lab1b/          # Multi-layer perceptrons  
├── lab2/           # RBF networks and SOMs
├── lab3/           # Hopfield networks
├── lab4/           # Deep Belief Networks and RBMs
└── README.md       # This file
```

Each lab directory contains:
- Implementation files (`.ipynb`, `.py`)
- Lab assignment descriptions (PDF)
- Final reports (PDF)
- Supporting data files
- Individual README with detailed information

## Usage

1. Navigate to the desired lab directory
2. Read the lab-specific README for detailed instructions
3. Open the Jupyter notebooks or Python files
4. Follow the implementation guidelines and complete the exercises

## Academic Context

This work was completed as part of the DD2437 course at KTH Royal Institute of Technology. The laboratories progress from fundamental concepts to state-of-the-art techniques, providing a comprehensive foundation in neural networks and deep learning.

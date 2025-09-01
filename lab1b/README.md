# Lab 1b: Multi-Layer Perceptrons

## Overview

This laboratory focuses on the implementation and comprehensive analysis of Multi-Layer Perceptrons (MLPs), representing a significant advancement from single-layer networks. Students implement MLPs from scratch and explore their application to both classification and function approximation tasks, including time series prediction.

## Objectives

- Implement Multi-Layer Perceptrons from scratch using NumPy
- Understand and implement the backpropagation algorithm
- Compare batch vs. sequential learning strategies
- Apply MLPs to classification and function approximation problems
- Explore time series prediction using PyTorch framework
- Investigate the effects of network architecture and hyperparameters

## Key Concepts Explored

### Neural Network Architecture
- **Multi-Layer Perceptrons**: Feedforward networks with hidden layers enabling non-linear learning
- **Activation Functions**: Sigmoid (tanh-like: `2/(1+exp(-x))-1`) and its derivative
- **Bias Units**: Essential components for shifting activation functions
- **Network Topology**: Input layer → Hidden layer → Output layer architecture

### Learning Algorithms
- **Forward Propagation**: Computing network outputs from inputs through layer-by-layer computation
- **Backward Propagation**: The fundamental algorithm for training neural networks
  - Error gradient computation through chain rule
  - Weight update calculation for all layers
- **Momentum**: Acceleration technique for gradient descent optimization

### Training Methodologies
- **Batch Learning**: Weight updates after processing entire dataset
- **Sequential (Online) Learning**: Weight updates after each individual sample
- **Optimization Algorithms**:
  - Gradient Descent (custom implementation)
  - Adam Optimizer (PyTorch implementation)
  - SGD with weight decay (L2 regularization)

### Advanced Concepts
- **Regularization**: L2 weight decay to prevent overfitting
- **Early Stopping**: Validation-based training termination
- **Learning Rate Scheduling**: Dynamic adjustment of learning parameters
- **Cross-Validation**: Proper model evaluation and hyperparameter tuning

## Implementation Details

### Files Structure
- `1b.ipynb`: Main notebook with MLP class and basic experiments
- `1b_part1.ipynb`: Detailed NumPy implementation with comprehensive analysis
- `1b_part2.ipynb`: PyTorch implementation for time series prediction
- `multiLayerPerceptron.py`: Standalone MLP class implementation

### NumPy Implementation (Custom)

#### Core MLP Class
```python
class MLP:
    def __init__(self, input_size, hidden_size, output_size)
    def epoch(self, inputs, targets)  # Single forward-backward pass
    def train(self, data, epochs)     # Full training loop
    def plot_decision_boundaries()    # Visualization utilities
```

#### Mathematical Implementation
- **Sigmoid Function**: `2/(1+exp(-x))-1` (tanh-like activation)
- **Forward Pass**: Layer-by-layer computation with bias terms
- **Backward Pass**: Error propagation using chain rule
- **Weight Updates**: Gradient descent with momentum support

#### Helper Functions
- `gen_data()`: Non-linearly separable 2D dataset generation
- `split_data()`: Various data splitting strategies for robustness testing
- `prepare_data()`: Data preprocessing for training
- `eval_data()`: Class-specific accuracy evaluation
- Function approximation utilities for 2D surface learning

### PyTorch Implementation (Modern Framework)

#### Neural Network Architecture
- Two hidden layers with configurable sizes
- Sigmoid activation for hidden layers
- Linear output layer for regression tasks
- MSE loss function for optimization

#### Training Framework
- Automated training loops with validation
- Early stopping mechanism
- Learning curve visualization
- Grid search for hyperparameter optimization

## Experimental Tasks

### Part 1: Classification with NumPy (Custom Implementation)

#### Task 1.1: Basic MLP Training
- Implement MLP with varying hidden unit counts (2, 5, 10, 20, 50, 100)
- Compare batch vs. sequential learning performance
- Analyze decision boundary complexity with different architectures

#### Task 1.2: Data Distribution Effects
- Test multiple data splitting scenarios:
  - Random 25% removal
  - 50% removal from specific class
  - Spatial region removal
- Evaluate generalization capabilities
- Analyze class-specific performance metrics

#### Task 1.3: Architecture Analysis
- Investigate the relationship between hidden units and model capacity
- Visualize decision boundaries for different network sizes
- Document overfitting vs. underfitting behaviors

### Part 2: Function Approximation (Custom Implementation)

#### Task 2.1: 2D Function Learning
- Approximate mathematical function: `z = exp(-(x²+y²)/10) - 0.5`
- Train MLPs with sequential learning
- Evaluate using MSE, RMSE, and MAE metrics
- Visualize learned surface vs. true function

#### Task 2.2: Architecture Impact
- Experiment with different hidden layer sizes
- Analyze approximation quality vs. network complexity
- Compare training and validation performance

### Part 3: Time Series Prediction (PyTorch Implementation)

#### Task 3.1: Mackey-Glass Series Prediction
- Generate chaotic Mackey-Glass time series
- Prepare temporal input-target pairs: `[x(t-20), x(t-15), x(t-10), x(t-5), x(t)] → x(t+5)`
- Implement sliding window approach for sequence processing

#### Task 3.2: Architecture Optimization
- Grid search over hidden layer configurations:
  - Hidden layer 1: [3, 4, 5] units
  - Hidden layer 2: [2, 4, 6] units
- Evaluate on test set using MSE
- Identify optimal architecture

#### Task 3.3: Regularization and Robustness
- Investigate L2 regularization (weight_decay) effects
- Test performance with noisy training data
- Analyze overfitting prevention techniques
- Compare training vs. validation loss curves

## Learning Outcomes

Upon completion, students will master:

### Theoretical Understanding
1. **Backpropagation Mathematics**: Complete derivation and implementation of gradient computation
2. **Universal Approximation**: Understanding MLPs' theoretical capacity for function approximation
3. **Optimization Dynamics**: How different learning strategies affect convergence
4. **Regularization Theory**: Mathematical basis for preventing overfitting

### Practical Skills
1. **From-Scratch Implementation**: Building neural networks using only fundamental libraries
2. **Framework Proficiency**: Effective use of PyTorch for complex tasks
3. **Hyperparameter Tuning**: Systematic approach to architecture optimization
4. **Performance Analysis**: Comprehensive evaluation methodologies

### Problem-Solving Applications
1. **Classification**: Non-linear decision boundary learning
2. **Regression**: Continuous function approximation
3. **Time Series**: Sequential data pattern recognition
4. **Optimization**: Balancing model complexity and generalization

## Technical Requirements

- Python 3.x
- NumPy for numerical computations
- Matplotlib for visualization
- PyTorch for deep learning framework
- Jupyter Notebook environment
- Optional: CUDA support for GPU acceleration

## Usage Instructions

### Getting Started
1. Begin with `1b.ipynb` for basic MLP concepts
2. Implement core forward and backward propagation
3. Experiment with different architectures and datasets

### Detailed Analysis
1. Work through `1b_part1.ipynb` for comprehensive NumPy implementation
2. Complete all classification and function approximation experiments
3. Document performance across different experimental conditions

### Advanced Applications
1. Open `1b_part2.ipynb` for PyTorch implementation
2. Generate and prepare Mackey-Glass time series data
3. Conduct grid search experiments
4. Analyze regularization effects on noisy data

### Custom Extensions
1. Modify `multiLayerPerceptron.py` for custom experiments
2. Implement additional activation functions
3. Experiment with different optimization algorithms

## Key Results and Insights

This laboratory demonstrates the power and versatility of multi-layer neural networks:

### Advantages Over Single-Layer Networks
- **Non-linear Learning**: Capability to learn complex decision boundaries
- **Universal Approximation**: Theoretical ability to approximate any continuous function
- **Flexible Architecture**: Scalable complexity for different problem domains

### Training Insights
- **Batch vs. Sequential**: Trade-offs between computational efficiency and convergence stability
- **Architecture Impact**: Optimal hidden unit count depends on problem complexity
- **Regularization Necessity**: Critical for preventing overfitting in complex models

### Practical Applications
- **Classification**: Superior performance on non-linearly separable data
- **Function Approximation**: Excellent results on smooth, continuous functions  
- **Time Series**: Effective pattern recognition in temporal data

This laboratory provides essential foundations for understanding modern deep learning architectures and prepares students for more advanced neural network concepts in subsequent labs.
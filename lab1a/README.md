# Lab 1a: Single-Layer Perceptrons and Delta Rule

## Overview

This laboratory introduces fundamental neural network concepts through the implementation of single-layer perceptrons for binary classification. Students explore the mathematical foundations of neural learning and investigate the limitations of linear models.

## Objectives

- Implement the Perceptron learning algorithm from scratch
- Implement the Delta rule (Widrow-Hoff rule) for gradient-based learning
- Understand linear separability and its implications for neural network performance
- Analyze the effects of data distribution and subsampling on learning performance
- Compare batch and sequential learning paradigms

## Key Concepts Explored

### Neural Network Fundamentals
- **Binary Classification**: Categorizing data points into two distinct classes
- **Linear Separability**: Understanding when classes can be perfectly separated by linear decision boundaries
- **Decision Boundaries**: Visualization and interpretation of learned classification regions

### Learning Algorithms
- **Perceptron Algorithm**: Classic algorithm for finding linear decision boundaries in linearly separable data
- **Delta Rule (Widrow-Hoff Rule/LMS)**: Gradient descent-based learning that minimizes Mean Squared Error
  - Sequential (Stochastic) version: Weight updates after each sample
  - Batch version: Weight updates after processing all samples

### Learning Dynamics
- **Weight Updates**: Mathematical formulation of parameter adjustments during training
- **Learning Rate**: Impact on convergence speed and stability
- **Epochs**: Complete passes through the training dataset
- **Mean Squared Error (MSE)**: Primary metric for evaluating model performance

### Data Analysis
- **Subsampling Effects**: Investigation of how different data removal strategies affect learning:
  - Random removal of training samples
  - Class-specific removal (removing samples from one class)
  - Spatial removal (removing samples from specific regions)
- **Generalization**: Model performance on unseen data
- **Evaluation Metrics**: Class-specific accuracy, sensitivity, and specificity

## Implementation Details

### Files Structure
- `1a.ipynb`: Main implementation notebook with Perceptron algorithm
- `part2.ipynb`: Delta rule implementation and comparative analysis

### Core Components

1. **Data Generation Functions**
   - `gen_data()`: Creates synthetic 2D datasets with controllable separability
   - `generate_base_dataset()`: Generates base distributions for experiments
   - `gen_non_lin_data()`: Creates non-linearly separable data for testing limitations
   - `subsample_data()`: Implements various subsampling strategies

2. **Learning Algorithms**
   - **Perceptron Implementation**: Binary threshold activation with weight updates on misclassified samples
   - **Delta Rule Implementation**: Continuous output with gradient-based weight updates
   - Support for both sequential and batch learning modes

3. **Visualization and Analysis**
   - `visualize_points()`: Plots data points with class labels
   - `plot_learning_rate()`: Displays learning curves (MSE over epochs)
   - Decision boundary visualization for both algorithms

4. **Evaluation Framework**
   - `eval_data()`: Calculates class-specific performance metrics
   - `evaluate_perceptron()`: Comprehensive model assessment
   - Statistical analysis of learning performance across different conditions

## Experimental Tasks

### Task 1: Basic Algorithm Implementation
- Implement both Perceptron and Delta rule algorithms
- Compare convergence behavior on linearly separable data
- Analyze the effect of learning rate on Delta rule performance

### Task 2: Learning Paradigm Comparison
- Compare sequential vs. batch learning for both algorithms
- Analyze convergence speed and final performance
- Investigate the smoothness of learning curves

### Task 3: Linear Separability Investigation
- Test both algorithms on linearly separable data
- Experiment with non-linearly separable datasets
- Document limitations of linear models

### Task 4: Data Distribution Analysis
- Conduct comprehensive subsampling experiments
- Test various removal strategies:
  - Random 25% removal
  - 50% removal from Class A
  - Spatial clustering effects
- Analyze impact on generalization performance

### Task 5: Performance Evaluation
- Calculate sensitivity and specificity for imbalanced scenarios
- Compare class-specific accuracy across different experimental conditions
- Statistical analysis of results across multiple runs

## Learning Outcomes

Upon completion, students will understand:

1. **Mathematical Foundations**: The mathematical basis of neural learning algorithms
2. **Algorithm Limitations**: Why linear models fail on non-linearly separable data
3. **Learning Dynamics**: How different update rules affect convergence behavior
4. **Data Quality Impact**: How data distribution affects model performance
5. **Evaluation Methodology**: Proper techniques for assessing classifier performance

## Technical Requirements

- Python 3.x
- NumPy for numerical computations
- Matplotlib for visualization
- Jupyter Notebook environment

## Usage Instructions

1. Open `1a.ipynb` to start with Perceptron implementation
2. Follow the guided exercises to implement core algorithms
3. Run experiments with different parameter settings
4. Proceed to `part2.ipynb` for Delta rule implementation
5. Complete the comparative analysis between algorithms
6. Document observations and analyze results

## Key Results and Insights

This lab demonstrates the fundamental principles of neural learning while highlighting the limitations of linear models. Students gain practical experience with:
- The mathematical elegance of simple learning rules
- The importance of data linearly separability
- The trade-offs between different learning paradigms
- The critical role of data quality in machine learning success

The foundation built in this lab is essential for understanding more complex neural architectures explored in subsequent laboratories.
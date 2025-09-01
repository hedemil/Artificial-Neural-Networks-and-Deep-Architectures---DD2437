# Lab 2: Radial Basis Function Networks and Self-Organizing Maps

## Overview

This laboratory explores two distinct neural network paradigms: Radial Basis Function (RBF) Networks for supervised learning and Self-Organizing Maps (SOMs) for unsupervised learning. Students implement both architectures from scratch and apply them to diverse problems including function approximation, classification, clustering, and combinatorial optimization.

## Objectives

- Implement Radial Basis Function Networks for regression and classification
- Understand and implement Self-Organizing Maps for unsupervised learning
- Compare batch vs. sequential learning in RBF networks
- Apply SOMs to clustering, visualization, and optimization problems
- Analyze the effects of hyperparameters on network performance

## Key Concepts Explored

### Radial Basis Function Networks

#### Architecture and Theory
- **RBF Network Structure**: Input layer → RBF hidden layer → Linear output layer
- **Gaussian Basis Functions**: `φ(x) = exp(-||x-μ||²/(2σ²))` with center `μ` and width `σ`
- **Localized Activation**: Each RBF unit responds maximally to inputs near its center
- **Linear Output Layer**: Simple weighted combination of RBF activations

#### Learning Paradigms
- **Batch Learning**: Direct analytical solution using pseudo-inverse
- **Sequential Learning**: Iterative gradient descent optimization
- **Weight Computation**: `w = (Φᵀ Φ)⁻¹ Φᵀ y` for batch method

#### Hyperparameter Analysis
- **RBF Width (σ)**: Controls the spread of basis functions
- **Number of RBF Units**: Affects model capacity and generalization
- **Center Placement**: Linear spacing vs. random initialization strategies
- **Noise Sensitivity**: Robustness to training data corruption

### Self-Organizing Maps (SOMs)

#### Unsupervised Learning Principles
- **Competitive Learning**: Winner-take-all mechanism for Best Matching Unit (BMU)
- **Neighborhood Function**: Gaussian updating of BMU neighbors
- **Topological Preservation**: Maintaining spatial relationships in low-dimensional mapping

#### Training Dynamics
- **BMU Selection**: `BMU = argmin ||x - wᵢ||²` (minimum Euclidean distance)
- **Weight Updates**: `wᵢ(t+1) = wᵢ(t) + α(t) h(i,BMU,t) [x - wᵢ(t)]`
- **Parameter Decay**: Exponential reduction of learning rate and neighborhood radius

#### Applications
- **Dimensionality Reduction**: High-dimensional data visualization
- **Clustering**: Unsupervised pattern discovery
- **Combinatorial Optimization**: Solving Traveling Salesperson Problem

## Implementation Details

### Files Structure
- `lab2_part1.ipynb`: RBF Networks implementation and analysis
- `lab2_part2.ipynb`: Self-Organizing Maps implementation and applications
- `data_lab2/`: Supporting datasets for experiments

### RBF Network Implementation

#### Core Functions
```python
def phi(x, mu, sigma):                    # Gaussian basis function
def phi_matrix(x, mu, sigma):            # RBF activation matrix
def predict(x, w, mu, sigma):            # Regression prediction
def predict_square(x, w, mu, sigma):     # Classification with sign function
def batch_learning(x, y, mu, sigma):     # Pseudo-inverse solution
def sequential_learning(x, y, mu, sigma, epochs, lr):  # Gradient descent
```

#### Analysis Functions
- `find_min_units_for_error()`: Determine minimum RBF units for target accuracy
- `analyze_rbf_width_effect()`: Systematic σ parameter investigation
- `analyze_node_positioning()`: Compare center placement strategies
- `compare_on_clean_data()`: Noise robustness evaluation

### Self-Organizing Map Implementation

#### SOM Class Architecture
```python
class SOM:
    def __init__(self, input_dim, output_shape):     # Initialize network
    def train(self, data, epochs):                   # Main training loop
    def find_best_matching_unit(self, x):            # BMU identification
    def update_weights(self, x, bmu_idx):            # Neighborhood updates
    def map_data_to_bmu(self, data):                 # Data mapping utilities
```

#### Specialized Configurations
- **1D SOMs**: Linear topology for animal clustering
- **Circular SOMs**: Ring topology for TSP optimization  
- **2D SOMs**: Grid topology for parliamentary voting analysis

## Experimental Tasks

### Part 1: RBF Network Analysis

#### Task 1.1: Function Approximation
- **Target Functions**:
  - Smooth: `sin(2x)` over specified domain
  - Discontinuous: Square wave function
- **Learning Comparison**: Batch vs. sequential methods
- **Performance Metrics**: MSE and MAE evaluation

#### Task 1.2: Hyperparameter Investigation
- **RBF Width Analysis**: Systematic variation of σ parameter
  - Wide σ: Smooth approximation, potential underfitting
  - Narrow σ: Detailed fitting, potential overfitting
- **Architecture Optimization**: Determine minimum units for accuracy thresholds
- **Center Placement**: Compare uniform spacing vs. random initialization

#### Task 1.3: Robustness Testing
- **Noise Impact**: Train on noisy data, test on clean data
- **Generalization**: Evaluate performance on unseen data regions
- **Learning Rate Effects**: Sequential learning parameter sensitivity

### Part 2: Self-Organizing Map Applications

#### Task 2.1: Animal Clustering
- **Dataset**: Animal attributes (32 animals × 84 features)
- **Objective**: Discover natural groupings based on characteristics
- **Implementation**: 1D SOM for linear organization
- **Analysis**: Visualize clustering and interpret biological relationships

#### Task 2.2: Traveling Salesperson Problem
- **Problem**: Find optimal tour through set of cities
- **Approach**: Circular SOM with ring topology
- **Algorithm**: 
  - Initialize circular network around city distribution
  - Train to minimize tour length
  - Extract tour from final node positions
- **Evaluation**: Compare solution quality with known optimal tours

#### Task 2.3: Political Data Visualization
- **Dataset**: Parliamentary voting records (MPs × votes)
- **Objective**: Visualize political landscape and party structures
- **Implementation**: 2D SOM for spatial organization
- **Visualization**: Color-code nodes by:
  - Political party affiliation
  - Gender distribution
  - Regional representation
- **Analysis**: Identify political clusters and voting patterns

## Learning Outcomes

Upon completion, students will understand:

### RBF Network Mastery
1. **Mathematical Foundation**: Gaussian basis functions and their properties
2. **Learning Algorithms**: Analytical vs. iterative optimization methods
3. **Hyperparameter Tuning**: Systematic approach to σ and architecture selection
4. **Robustness Analysis**: Understanding noise effects and generalization

### SOM Expertise  
1. **Unsupervised Learning**: Competitive learning and self-organization principles
2. **Topological Mapping**: Preserving data structure in lower dimensions
3. **Parameter Dynamics**: Learning rate and neighborhood scheduling
4. **Application Versatility**: From clustering to optimization problems

### Comparative Analysis
1. **Supervised vs. Unsupervised**: Understanding different learning paradigms
2. **Local vs. Global**: RBF localization vs. SOM global organization
3. **Architecture Selection**: Choosing appropriate methods for specific problems
4. **Performance Evaluation**: Metrics and validation strategies for different tasks

## Technical Requirements

- Python 3.x
- NumPy for numerical computations
- Matplotlib for visualization and analysis
- Jupyter Notebook environment
- Data files in `data_lab2/` directory

## Usage Instructions

### RBF Network Experiments
1. Open `lab2_part1.ipynb`
2. Implement core RBF functions following mathematical specifications
3. Run function approximation experiments on `sin(2x)` and square wave
4. Complete hyperparameter analysis with different σ values
5. Test noise robustness and center placement strategies

### SOM Applications
1. Open `lab2_part2.ipynb`
2. Implement SOM class with proper initialization and training
3. Load animal dataset and run clustering experiment
4. Implement TSP solver using circular SOM topology
5. Analyze political voting data with 2D SOM visualization

### Advanced Experiments
1. Modify hyperparameters to explore different behaviors
2. Test on custom datasets to validate generalization
3. Implement additional visualization techniques
4. Compare results across different parameter settings

## Key Results and Insights

### RBF Network Findings
- **Batch Learning**: Provides optimal solution but requires matrix inversion
- **Sequential Learning**: More flexible but requires careful learning rate tuning  
- **Width Parameter**: Critical for balancing approximation quality and generalization
- **Center Placement**: Uniform spacing often superior to random initialization

### SOM Discoveries
- **Animal Clustering**: Reveals natural taxonomic relationships
- **TSP Optimization**: Demonstrates SOM versatility beyond clustering
- **Political Analysis**: Uncovers party structures and ideological distances
- **Topology Preservation**: Maintains meaningful spatial relationships

### Methodological Insights
- **Problem-Method Matching**: Different tasks benefit from different approaches
- **Parameter Sensitivity**: Both methods require careful hyperparameter tuning
- **Scalability**: SOMs handle high-dimensional data more gracefully
- **Interpretability**: Both methods provide interpretable results and visualizations

This laboratory provides essential experience with both supervised and unsupervised neural learning, preparing students for advanced topics in neural networks and machine learning.
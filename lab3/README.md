# Lab 3: Hopfield Networks

## Overview

This laboratory focuses on the implementation and analysis of Hopfield Networks, a class of recurrent neural networks that serve as associative memory systems. Students explore the mathematical foundations, energy dynamics, and storage capacity of these networks while investigating their ability to retrieve patterns from partial or noisy inputs.

## Objectives

- Implement Hopfield Networks from scratch with complete mathematical foundations
- Understand associative memory principles and pattern storage mechanisms
- Analyze network stability through energy function dynamics
- Investigate storage capacity limits and factors affecting performance
- Explore sparse pattern representation and specialized update rules

## Key Concepts Explored

### Network Architecture and Theory
- **Recurrent Architecture**: Fully connected network where every neuron connects to every other
- **Symmetric Weights**: Ensuring convergence through weight symmetry (W_ij = W_ji)
- **No Self-Connections**: Diagonal weight matrix elements set to zero (W_ii = 0)
- **Binary States**: Neurons with discrete states {-1, +1} or {0, 1}

### Learning and Storage
- **Hebbian Learning Rule**: `W_ij = (1/N) Σ_p x_i^p x_j^p` for pattern storage
- **Associative Memory**: Ability to retrieve complete patterns from partial inputs
- **Pattern Storage**: Encoding multiple patterns in connection weights
- **Storage Capacity**: Theoretical limit of ~0.138N patterns for N neurons

### Network Dynamics
- **Energy Function**: `E = -½ Σ_i Σ_j W_ij x_i x_j` (Lyapunov function)
- **Stability Analysis**: Energy minimization guarantees convergence
- **Attractors**: Stable states corresponding to stored patterns
- **Spurious States**: Undesired stable states not corresponding to stored patterns

### Update Mechanisms
- **Synchronous Update**: All neurons update simultaneously
- **Asynchronous Update**: Sequential random neuron updates
- **Convergence**: Network evolution toward stable attractor states

## Implementation Details

### Files Structure
- `lab3.ipynb`: Main implementation notebook with comprehensive experiments
- `hopfield.py`: Standalone Hopfield network class implementation
- `pict.dat`: Binary image data for pattern storage experiments

### Core Hopfield Network Implementation

#### Network Class Structure
```python
class Hopfield:
    def __init__(self, N):                              # Initialize N-neuron network
    def train(self, patterns, **kwargs):                # Store patterns using Hebbian rule
    def recall(self, pattern, **kwargs):                # Pattern retrieval with dynamics
    def calc_energy(self, pattern):                     # Energy function evaluation
    def is_stable(self, pattern):                       # Stability verification
```

#### Training Methods
- **Standard Hebbian**: `W += outer_product(pattern, pattern.T)`
- **Random Weights**: Alternative initialization for comparison
- **Symmetric Enforcement**: `W = 0.5 * (W + W.T)` for guaranteed convergence
- **Sparse Patterns**: Modified rule incorporating pattern sparsity

#### Recall Algorithms
- **Synchronous Recall**: `x_new = sign(W @ x_old)`
- **Asynchronous Recall**: Sequential single-neuron updates
- **Energy Tracking**: Optional energy evolution recording
- **Convergence Detection**: Automatic stopping when stable state reached

### Advanced Features

#### Sparse Hopfield Networks
- **Sparsity Parameter (ρ)**: Mean activity level in sparse patterns
- **Threshold (θ)**: Activation threshold for sparse pattern processing  
- **Modified Learning**: Adapted Hebbian rule for sparse representations
- **Enhanced Capacity**: Improved storage for sparse pattern sets

#### Analysis Utilities
- **Pattern Generation**: Tools for creating test patterns
- **Hamming Distance**: Quantifying pattern similarity and retrieval accuracy
- **Visualization**: Display functions for 32×32 binary images
- **Performance Metrics**: Success rate calculation and statistical analysis

## Experimental Tasks

### Part 1: Basic Hopfield Network (N=8)

#### Task 1.1: Small Network Analysis
- **Pattern Storage**: Store 3 simple binary patterns
- **Degraded Recall**: Test retrieval with 1-2 bit errors
- **Convergence Verification**: Confirm stable state achievement
- **Energy Dynamics**: Track energy evolution during recall

#### Task 1.2: Complete Attractor Analysis  
- **Exhaustive Search**: Test all 2^N possible states
- **Attractor Identification**: Find all stable states in network
- **Spurious States**: Identify and analyze undesired attractors
- **Basin of Attraction**: Map states that converge to each attractor

#### Task 1.3: Robustness Testing
- **High Noise**: Test patterns with >50% corrupted bits
- **Failure Modes**: Identify conditions leading to incorrect recall
- **Distance Effects**: Relate initial Hamming distance to success probability

### Part 2: Image Pattern Storage (N=1024)

#### Task 2.1: High-Dimensional Pattern Storage
- **Image Loading**: Process 32×32 binary images from `pict.dat`
- **Pattern Training**: Store 3 distinct image patterns
- **Stability Verification**: Confirm stored patterns are stable states
- **Noise Robustness**: Test recall with 100 random bit flips

#### Task 2.2: Pattern Completion
- **Degraded Images**: Use provided partial patterns (p10, p11)
- **Completion Quality**: Evaluate reconstruction accuracy
- **Visual Assessment**: Compare original and retrieved images
- **Failure Analysis**: Identify patterns that cannot be completed

#### Task 2.3: Update Rule Comparison
- **Synchronous vs. Asynchronous**: Compare convergence behavior
- **Step-by-Step Evolution**: Visualize asynchronous recall process
- **Convergence Speed**: Measure iterations to stability
- **Quality Differences**: Assess final state accuracy for each method

### Part 3: Energy Landscape Analysis

#### Task 3.1: Energy Evolution Tracking
- **Stored Pattern Dynamics**: Monitor energy during recall of original patterns
- **Energy Minimization**: Verify monotonic decrease property
- **Convergence Patterns**: Analyze energy trajectory characteristics

#### Task 3.2: Degraded Pattern Energy
- **Noisy Initial States**: Track energy evolution from corrupted patterns
- **Recovery Paths**: Visualize energy landscape navigation
- **Local Minima**: Identify convergence to spurious states

#### Task 3.3: Random Weight Analysis
- **Random Initialization**: Compare with Hebbian-trained networks
- **Symmetric vs. Asymmetric**: Effect of weight symmetry on convergence
- **Energy Behavior**: Analyze energy evolution with random weights

### Part 4: Capacity and Robustness Analysis

#### Task 4.1: Noise Robustness Quantification
- **Systematic Noise Addition**: 0% to 100% bit corruption
- **Hamming Distance Measurement**: Quantify retrieval accuracy
- **Robustness Curves**: Plot accuracy vs. noise level
- **Critical Noise Threshold**: Identify performance breakdown points

#### Task 4.2: Storage Capacity Investigation
- **Incremental Loading**: Add patterns until capacity exceeded
- **Success Rate Measurement**: Fraction of correctly recalled patterns
- **Capacity Curves**: Plot performance vs. number of stored patterns
- **Theoretical Comparison**: Validate against 0.138N theoretical limit

#### Task 4.3: Self-Connection Analysis
- **With/Without Self-Connections**: Compare W_ii = 0 vs. W_ii ≠ 0
- **Capacity Impact**: Effect on maximum storage capacity
- **Stability Changes**: Influence on attractor dynamics

### Part 5: Sparse Hopfield Networks

#### Task 5.1: Sparse Pattern Generation
- **Controlled Sparsity**: Generate patterns with specified activity levels
- **Sparsity Parameter (ρ)**: Vary from dense to highly sparse patterns
- **Pattern Diversity**: Ensure sufficient variation in sparse patterns

#### Task 5.2: Sparse Network Training
- **Modified Hebbian Rule**: Implement sparsity-aware learning
- **Threshold Integration**: Incorporate activation thresholds
- **Capacity Enhancement**: Measure improved storage for sparse patterns

#### Task 5.3: Threshold Optimization
- **Parameter Sweeps**: Systematic variation of threshold (θ)
- **Success Rate Analysis**: Optimal threshold identification
- **Sparsity Dependencies**: Threshold requirements vs. pattern sparsity

## Learning Outcomes

Upon completion, students will master:

### Theoretical Understanding
1. **Associative Memory Principles**: How networks store and retrieve patterns
2. **Energy Function Theory**: Mathematical basis for network stability
3. **Capacity Limitations**: Theoretical and practical storage bounds
4. **Convergence Guarantees**: Conditions ensuring stable state achievement

### Mathematical Foundations
1. **Hebbian Learning**: Mathematical formulation and biological motivation
2. **Lyapunov Functions**: Energy-based stability analysis
3. **Linear Algebra**: Matrix operations and eigenvalue relationships
4. **Probability Theory**: Statistical analysis of network performance

### Implementation Skills
1. **Recurrent Network Programming**: Complete implementation from mathematical specifications
2. **Algorithm Optimization**: Efficient update rule implementations
3. **Data Analysis**: Statistical evaluation of network performance
4. **Visualization**: Effective presentation of high-dimensional results

### Problem-Solving Applications
1. **Pattern Recognition**: Robust retrieval from noisy inputs
2. **Associative Recall**: Content-addressable memory systems
3. **Optimization**: Understanding energy minimization approaches
4. **Capacity Planning**: Designing networks for specific storage requirements

## Technical Requirements

- Python 3.x
- NumPy for numerical computations and linear algebra
- Matplotlib for visualization and plotting
- Jupyter Notebook environment
- Binary image data files (`pict.dat`)

## Usage Instructions

### Getting Started
1. Open `lab3.ipynb` and review the Hopfield class structure
2. Implement core functions following mathematical specifications
3. Start with small network (N=8) experiments for verification

### Progressive Experiments
1. **Small Network**: Complete Part 1 tasks for fundamental understanding
2. **Image Storage**: Scale to N=1024 for realistic pattern storage
3. **Energy Analysis**: Implement energy tracking and visualization
4. **Capacity Studies**: Conduct systematic capacity investigations

### Advanced Analysis
1. **Sparse Networks**: Implement sparse pattern modifications
2. **Parameter Optimization**: Find optimal thresholds and learning parameters
3. **Custom Experiments**: Design additional tests for specific hypotheses

### Performance Evaluation
1. Run multiple trials for statistical significance
2. Compare results with theoretical predictions
3. Document failure modes and limiting factors
4. Validate implementation against known benchmarks

## Key Results and Insights

### Network Behavior
- **Convergence Reliability**: Synchronous updates guarantee convergence
- **Energy Minimization**: Consistent energy decrease during recall
- **Attractor Dynamics**: Multiple stable states with defined basins
- **Noise Tolerance**: Graceful degradation with increasing corruption

### Capacity Findings
- **Theoretical Limits**: ~0.138N capacity validated experimentally  
- **Sparse Advantages**: Higher effective capacity with sparse patterns
- **Self-Connection Impact**: Diagonal weights reduce storage capacity
- **Pattern Correlation**: Orthogonal patterns store more reliably

### Practical Applications
- **Associative Memory**: Effective content-addressable storage
- **Error Correction**: Natural robustness to input corruption
- **Pattern Completion**: Successful reconstruction from partial inputs
- **Optimization**: Energy minimization for computational problems

This laboratory provides deep insight into recurrent neural networks and associative memory, establishing foundations for understanding more complex dynamical systems and their applications in neural computation.
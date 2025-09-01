# Lab 4: Deep Belief Networks and Restricted Boltzmann Machines

## Overview

This laboratory focuses on implementing Deep Belief Networks (DBNs) built from Restricted Boltzmann Machines (RBMs), representing a significant advancement in deep generative modeling. Students implement these energy-based models from scratch and apply them to the MNIST handwritten digit dataset for both classification and generation tasks.

## Objectives

- Implement Restricted Boltzmann Machines (RBMs) from scratch
- Build Deep Belief Networks through layer-wise stacking of RBMs  
- Understand Contrastive Divergence training algorithm
- Apply DBNs to handwritten digit recognition and generation
- Explore unsupervised pre-training and fine-tuning strategies

## Key Concepts Explored

### Energy-Based Models
- **Energy Functions**: Joint probability distributions defined by energy landscapes
- **Partition Functions**: Normalization constants for probability distributions
- **Gibbs Distributions**: `P(x) = exp(-E(x))/Z` relationship between energy and probability
- **Maximum Likelihood Learning**: Training through gradient-based likelihood maximization

### Restricted Boltzmann Machines
- **Bipartite Architecture**: Visible and hidden units with no intra-layer connections
- **Binary Stochastic Units**: Probabilistic binary activations
- **Conditional Independence**: Hidden units independent given visible states (and vice versa)
- **Weight Symmetry**: Bidirectional connections with identical weights

### Deep Belief Networks  
- **Hierarchical Feature Learning**: Layer-wise representation building
- **Greedy Pre-training**: Sequential training of individual RBM layers
- **Generative vs. Discriminative**: Dual-purpose architecture for both tasks
- **Directed vs. Undirected**: Mixed architecture with directed connections

### Training Algorithms
- **Contrastive Divergence (CD-k)**: Approximate maximum likelihood training
- **Gibbs Sampling**: Alternating conditional sampling for inference
- **Wake-Sleep Algorithm**: Optional fine-tuning for entire network
- **Gradient Approximation**: Efficient learning without intractable partition function

## Implementation Details

### Files Structure
- `run.py`: Main execution script with experiments and analysis
- `rbm.py`: Restricted Boltzmann Machine implementation
- `dbn.py`: Deep Belief Network architecture and training
- `util.py`: Utility functions for data processing and visualization
- `README.txt`: Detailed task specifications and implementation guidance
- MNIST dataset files: Training and test data for digit recognition

### RBM Implementation (`rbm.py`)

#### Core RBM Class
```python
class RestrictedBoltzmannMachine:
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10)
    def cd1(self, visible_trainset, n_iterations=1000)           # Contrastive Divergence training
    def get_h_given_v(self, visible_minibatch)                   # P(h|v) inference
    def get_v_given_h(self, hidden_minibatch)                    # P(v|h) inference  
    def get_h_given_v_dir(self, visible_minibatch)               # Directed inference
    def get_v_given_h_dir(self, hidden_minibatch)                # Directed inference
```

#### Mathematical Foundations
- **Conditional Probabilities**: 
  - `P(h_j=1|v) = σ(c_j + Σ_i W_ij v_i)`
  - `P(v_i=1|h) = σ(b_i + Σ_j W_ij h_j)`
- **Contrastive Divergence**: Approximate gradient using finite Gibbs chains
- **Weight Updates**: `ΔW ∝ <vh>_data - <vh>_model`

#### Special Handling
- **Bottom RBM**: Direct connection to visible input data
- **Top RBM**: Associative layer combining features and labels
- **Label Integration**: Softmax activation for classification outputs

### DBN Implementation (`dbn.py`)

#### Network Architecture
```python
class DeepBeliefNet:
    def __init__(self, sizes, image_size, n_labels, batch_size)
    def train_greedylayerwise(self, vis_trainset, n_iterations)  # Layer-wise pre-training
    def recognize(self, true_img)                                # Classification inference
    def generate(self, true_label, n_sample_gibbs)               # Image generation
    def train_wakesleep_finetune(self, vis_trainset, n_iterations) # Fine-tuning
```

#### Training Strategy
- **Layer 1**: Train bottom RBM on raw image data
- **Layer 2**: Train middle RBM on Layer 1 hidden representations  
- **Layer 3**: Train top associative RBM on Layer 2 features + labels
- **Architecture**: `[vis] ↔ [hid] → [pen] ↔ [top+labels]`

#### Inference Modes
- **Recognition**: Bottom-up feature extraction followed by classification
- **Generation**: Top-down sampling from label-conditioned model
- **Bidirectional**: Full generative model with both directions

## Experimental Tasks

### Task 4.1: RBM Implementation

#### Subtask 4.1.1: Forward and Backward Passes
- **Implement `get_h_given_v()`**: Compute hidden unit probabilities from visible units
  - Calculate pre-activations: `h_pre = c + W^T @ v`
  - Apply sigmoid: `p_h = σ(h_pre)`
  - Sample binary states: `h = Bernoulli(p_h)`

- **Implement `get_v_given_h()`**: Compute visible unit probabilities from hidden units
  - Calculate pre-activations: `v_pre = b + W @ h`
  - Apply activation (sigmoid for standard, softmax for labels)
  - Sample appropriate distributions

#### Subtask 4.1.2: Parameter Updates
- **Implement `update_params()`**: Calculate and apply gradients
  - **Positive Phase**: `<vh>_data` from data samples
  - **Negative Phase**: `<vh>_model` from model samples
  - **Weight Update**: `W += learning_rate * (positive - negative)`
  - **Bias Updates**: Separate updates for visible and hidden biases

#### Subtask 4.1.3: Contrastive Divergence
- **Implement `cd1()`**: Complete CD-1 training algorithm
  - Initialize visible units with training data
  - Perform one step of Gibbs sampling
  - Calculate positive and negative statistics
  - Update parameters using computed gradients
  - Track reconstruction error and learning progress

### Task 4.2: DBN Construction and Training

#### Subtask 4.2.1: Greedy Layer-wise Training
- **Implement `train_greedylayerwise()`**: Sequential RBM training
  - Train bottom RBM on image data
  - Extract hidden representations from trained RBM
  - Use representations as visible data for next RBM
  - Continue until all layers trained
  - Handle label integration in top layer

#### Subtask 4.2.2: Recognition Implementation  
- **Implement `recognize()`**: Classification through DBN
  - Forward pass: propagate image through trained layers
  - Top-layer inference: run Gibbs sampling in associative RBM
  - Label extraction: read classification from label units
  - Confidence assessment: analyze label probabilities

#### Subtask 4.2.3: Generation Implementation
- **Implement `generate()`**: Image synthesis from labels  
  - Clamp desired label in top layer
  - Sample from label-conditioned distribution
  - Backward pass: propagate down through network layers
  - Image reconstruction: generate pixel values from features

#### Subtask 4.2.4: Directed Inference
- **Implement directed versions**: `get_h_given_v_dir()` and `get_v_given_h_dir()`
  - Use recognition weights for bottom-up inference
  - Use generative weights for top-down inference
  - Handle asymmetric connections in stacked architecture

### Task 4.3: Wake-Sleep Fine-tuning (Optional)

#### Subtask 4.3.1: Wake-Sleep Algorithm
- **Implement `train_wakesleep_finetune()`**: Full network fine-tuning
  - **Wake Phase**: Bottom-up recognition with generative weight updates
  - **Sleep Phase**: Top-down generation with recognition weight updates
  - **Alternating Updates**: Balance generative and recognition capabilities

#### Subtask 4.3.2: Parameter Update Rules
- **Implement `update_recognize_params()`**: Recognition weight updates during sleep
- **Implement `update_generate_params()`**: Generative weight updates during wake
- **Bidirectional Training**: Simultaneous optimization of both pathways

## Learning Outcomes

Upon completion, students will master:

### Theoretical Foundations
1. **Energy-Based Modeling**: Understanding probability distributions through energy functions
2. **Unsupervised Learning**: Feature learning without explicit supervision
3. **Generative Modeling**: Creating new data samples from learned distributions
4. **Approximate Inference**: Practical solutions for intractable probabilistic models

### Deep Learning Concepts
1. **Layer-wise Pre-training**: Historical importance in deep learning development
2. **Representation Learning**: Automatic feature discovery through multiple layers
3. **Generative vs. Discriminative**: Dual-purpose architectures for multiple tasks
4. **Probabilistic Networks**: Stochastic neural network architectures

### Implementation Expertise
1. **From-Scratch Development**: Building complex models from mathematical foundations
2. **Probabilistic Programming**: Implementing stochastic neural networks
3. **Sampling Algorithms**: Gibbs sampling and Monte Carlo methods
4. **Optimization Techniques**: Approximate gradient methods for complex objectives

### Practical Applications
1. **Handwritten Digit Recognition**: State-of-the-art classification performance
2. **Image Generation**: Creating realistic synthetic images
3. **Feature Visualization**: Understanding learned representations
4. **Anomaly Detection**: Identifying unusual patterns through reconstruction

## Technical Requirements

- Python 3.x
- NumPy for numerical computations and probability operations
- Matplotlib for visualization and result analysis
- MNIST dataset (provided in binary format)
- Sufficient computational resources for iterative training

## Usage Instructions

### Getting Started
1. Review `README.txt` for detailed task specifications
2. Examine existing code structure in `rbm.py` and `dbn.py`
3. Start with RBM implementation in Task 4.1
4. Test implementations with small-scale experiments

### RBM Development
1. Implement forward and backward inference functions
2. Complete parameter update calculations
3. Integrate components into CD-1 algorithm
4. Validate with reconstruction error monitoring

### DBN Construction  
1. Build on tested RBM implementation
2. Implement greedy layer-wise training strategy
3. Add recognition and generation capabilities
4. Test on MNIST digit classification and synthesis

### Advanced Features
1. Implement directed inference for stacked architecture
2. Add wake-sleep fine-tuning capabilities
3. Experiment with different architectures and hyperparameters
4. Analyze learned representations and generated samples

### Performance Evaluation
1. Monitor training convergence through reconstruction error
2. Evaluate classification accuracy on MNIST test set
3. Assess generation quality through visual inspection
4. Compare with baseline methods and literature results

## Key Results and Insights

### RBM Behavior
- **Feature Learning**: Automatic discovery of edge detectors and digit parts
- **Reconstruction Quality**: Faithful reproduction of training examples
- **Generative Capability**: Synthesis of novel digit-like images
- **Training Dynamics**: Convergence patterns and stability issues

### DBN Performance
- **Classification Accuracy**: Competitive results on MNIST benchmark
- **Hierarchical Features**: Layer-wise representation complexity
- **Generation Quality**: Realistic synthetic digit images
- **Unsupervised Pre-training**: Benefits for subsequent supervised tasks

### Historical Significance
- **Deep Learning Revival**: Role in renewed interest in neural networks
- **Feature Learning**: Automatic discovery without hand-crafted features
- **Scalability**: Path toward deeper architectures
- **Theoretical Framework**: Principled approach to unsupervised learning

### Modern Relevance
- **Generative Models**: Foundation for modern generative architectures
- **Representation Learning**: Core concept in contemporary deep learning
- **Energy-Based Models**: Continued relevance in probabilistic modeling
- **Probabilistic Reasoning**: Framework for uncertainty in neural networks

This laboratory provides essential insight into the historical development of deep learning while teaching fundamental concepts that remain relevant in modern neural network architectures. Students gain both theoretical understanding and practical implementation experience with generative models and deep networks.
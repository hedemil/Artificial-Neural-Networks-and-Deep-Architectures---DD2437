from util import *
import math
from tqdm import tqdm

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10, lr=0.01, momentum=0.7):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))
        # self.bias_v = np.zeros(self.ndim_visible)

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        # self.bias_h = np.zeros(self.ndim_hidden)
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = lr
        
        self.momentum = momentum

        self.weight_decay = 0.0001

        self.print_period = 2
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 1, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }

        self.losses = {'epoch': [], 'recon_loss': []}
        self.errors = []
        self.weights_vh = {'epoch': [], 'weights': []}
        self.weight_updates = []


        
        return

        
    def cd1(self,visible_trainset, n_epochs=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]
        num_batches = math.ceil(n_samples / self.batch_size)
        train_batches = np.array_split(visible_trainset, num_batches)

        for epoch in tqdm(range(n_epochs)):
            epoch_losses = []  # list to collect batch losses
            np.random.shuffle(train_batches)
            
            for i, batch in enumerate(train_batches):
                # Positive phase: use binary sampling for data-driven hidden activations.
                v_0 = batch  # Real mini-batch data
                prob_h_0, h_0 = self.get_h_given_v(v_0)

                # Negative phase: when driven by reconstructions, use probabilities without sampling.
                prob_v_1, v_1 = self.get_v_given_h(h_0)
                prob_h_1, h_1 = self.get_h_given_v(v_1)

                # Update parameters using probabilities (not binary states)
                self.update_params(v_0, h_0, v_1, h_1)

                # Compute the reconstruction error for this batch
                error = np.sum((v_0 - v_1)**2) / v_0.shape[0]
                epoch_losses.append(error)

            # Compute the average reconstruction loss for the entire epoch
            avg_epoch_loss = np.mean(epoch_losses)
            
            # Optionally print the average loss every print_period epochs
            if epoch % self.print_period == 0:
                print("Epoch=%d, Average Recon Loss=%4.4f" % (epoch, avg_epoch_loss))
                print(f'Weight max: {np.max(self.weight_vh)}')
                print(f'Weight min: {np.min(self.weight_vh)}')
                
            if epoch % self.rf["period"] == 0 and self.is_bottom:
                
                viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=epoch, grid=self.rf["grid"])
            
            # Save the average loss for plotting later
            self.losses['epoch'].append(epoch)
            self.losses['recon_loss'].append(avg_epoch_loss)
            
        return
    

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        
        # Compute gradients
        grad_bias_v = np.mean(v_0 - v_k, axis=0)
        grad_weight_vh = (np.dot(v_0.T, h_0) - np.dot(v_k.T, h_k)) / v_0.shape[0]
        grad_bias_h = np.mean(h_0 - h_k, axis=0)

        # Update deltas using momentum, learning rate, and weight decay
        self.delta_bias_v = self.momentum * self.delta_bias_v + self.learning_rate * grad_bias_v
        self.delta_bias_h = self.momentum * self.delta_bias_h + self.learning_rate * grad_bias_h

        # No weight decay
        self.delta_weight_vh = self.momentum * self.delta_weight_vh + self.learning_rate * grad_weight_vh

        # # Apply weight decay to the weight updates
        # self.delta_weight_vh = self.momentum * self.delta_weight_vh + self.learning_rate * (grad_weight_vh - self.weight_decay * self.weight_vh)

        # # No momentum or weight decay
        # self.delta_bias_v = self.learning_rate * grad_bias_v
        # self.delta_weight_vh = self.learning_rate * grad_weight_vh
        # self.delta_bias_h = self.learning_rate * grad_bias_h

        # Update parameters
        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h

        self.weight_updates.append(np.sum(self.delta_weight_vh))
        
        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 
        
        hidden_activations = np.dot(visible_minibatch, self.weight_vh) + self.bias_h
        hidden_prob = sigmoid(hidden_activations)
        hidden_states = sample_binary(hidden_prob)

        return hidden_prob, hidden_states


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            support = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            visible_prob = np.ndarray(shape=support.shape)
            visible_states = np.ndarray(shape=support.shape)

            # Split into data and label parts
            data_support = support[:, :-self.n_labels]
            label_support = support[:, -self.n_labels:]

            # Compute probabilities and activations for data and labels
            data_prob = sigmoid(data_support)
            data_states = sample_binary(data_prob)
            label_prob = softmax(label_support)
            label_states = sample_categorical(label_prob)

            # Concatenate data and labels
            visible_prob = np.concatenate((data_prob, label_prob), axis=1)
            visible_states = np.concatenate((data_states, label_states), axis=1)
            
        else:
                        
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)             
            visible_activation = np.dot(hidden_minibatch, self.weight_vh.T) + self.bias_v
            visible_prob = sigmoid(visible_activation)
            visible_states = sample_binary(visible_prob)

        return visible_prob, visible_states


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below)
        # Compute probabilities and activations of hidden layer
        hidden_activations = np.dot(visible_minibatch, self.weight_v_to_h) + self.bias_h
        hidden_prob = sigmoid(hidden_activations)
        hidden_states = sample_binary(hidden_prob) 
        
        return hidden_prob, hidden_states

    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            
            # Set probabilities and states to zeros
            visible_prob = np.zeros((n_samples,self.ndim_visible))
            visible_states = np.zeros((n_samples,self.ndim_visible))

            # Raise error
            raise ValueError("Top RBM should not have directed connections.")
            
        else:
                        
            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            # Compute probabilities and activations of visible layer
            visible_activation = np.dot(hidden_minibatch, self.weight_h_to_v) + self.bias_v
            visible_prob = sigmoid(visible_activation)
            visible_states = sample_binary(visible_prob)
            
        return visible_prob, visible_states    
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    

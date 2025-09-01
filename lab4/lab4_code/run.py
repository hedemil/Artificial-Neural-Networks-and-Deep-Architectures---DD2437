from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def plot_mean_label_evolution(label_snapshots):
    """
    Plots how the *mean* label distribution across the entire batch evolves
    over multiple Gibbs iterations in the top RBM.

    Args:
        label_snapshots: List of length n_gibbs_recog. 
                         Each element is shape (batch_size, n_labels).
    """
    label_snapshots_array = np.array(label_snapshots)  # (n_iterations, batch_size, n_labels)
    n_iterations = label_snapshots_array.shape[0]
    n_labels = label_snapshots_array.shape[2]
    
    # Compute mean label probabilities at each iteration: shape (n_iterations, n_labels)
    mean_label_prob = label_snapshots_array.mean(axis=1)  
    
    plt.figure(figsize=(7,5))
    for lbl_idx in range(n_labels):
        plt.plot(
            range(n_iterations), 
            mean_label_prob[:, lbl_idx], 
            label=f'Label {lbl_idx}'
        )
    
    plt.xlabel("Gibbs Iteration")
    plt.ylabel("Mean Probability Across Batch")
    plt.title("Mean Label Distribution Evolution")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    # Max 60k training images and 10k test images

    print("\nStarting Restricted Boltzmann Machines..")

    hidden_configs = [200, 500]  # different hidden layer sizes
    
    convergence = {}
    rbm_models = {}

    # lr_configs = [0.01, 0.05]
    # momentum_configs = [0.7, 0.9]

    

    # for lr in lr_configs:
    #     convergence[lr] = {}    # initialize nested dictionary for each lr
    #     rbm_models[lr] = {}
    #     for momentum in momentum_configs:
    #         print(f"Running RBM with learning rate = {lr}, momentum = {momentum}")
    #         rbm = RestrictedBoltzmannMachine(
    #             ndim_visible=image_size[0]*image_size[1],
    #             ndim_hidden=500,
    #             is_bottom=True,
    #             image_size=image_size,
    #             is_top=False,
    #             n_labels=10,
    #             batch_size=20,
    #             lr=lr,
    #             momentum=momentum
    #         )
            
    #         # Run training and store convergence metrics
    #         rbm.cd1(visible_trainset=train_imgs, n_epochs=20)
            
    #         # Save the metrics for reconstruction loss vs. epoch
    #         convergence[lr][momentum] = {
    #             'epochs': rbm.losses['epoch'],
    #             'recon_loss': rbm.losses['recon_loss']
    #         }
    #         # Also store the trained model itself for later plots
    #         rbm_models[lr][momentum] = rbm
    
    # # Create subplots for each learning rate
    # fig, axes = plt.subplots(nrows=1, ncols=len(lr_configs), figsize=(15, 5), sharey=True)

    # for i, lr in enumerate(lr_configs):
    #     ax = axes[i]
    #     for momentum in momentum_configs:
    #         epochs = convergence[lr][momentum]['epochs']
    #         recon_loss = convergence[lr][momentum]['recon_loss']
    #         ax.plot(epochs, recon_loss, label=f"mom={momentum}")
    #     ax.set_xlabel("Epoch")
    #     ax.set_title(f"lr={lr}")
    #     ax.legend(loc="best")
    #     if i == 0:
    #         ax.set_ylabel("Reconstruction Loss")

    # plt.suptitle("Reconstruction Loss Convergence for Each Hyperparameter Combination")
    # plt.tight_layout()
    # plt.show()

    # for ndim in hidden_configs:
    #     print(f"Running RBM with hidden layer size = {ndim}")
    #     rbm = RestrictedBoltzmannMachine(
    #         ndim_visible=image_size[0]*image_size[1],
    #         ndim_hidden=ndim,
    #         is_bottom=True,
    #         image_size=image_size,
    #         is_top=False,
    #         n_labels=10,
    #         batch_size=10
    #     )
        
    #     # Run training and store convergence metrics
    #     rbm.cd1(visible_trainset=train_imgs, n_epochs=20)
        
    #     # Save the metrics for reconstruction loss vs. epoch
    #     convergence[ndim] = {
    #         'epochs': rbm.losses['epoch'],
    #         'recon_loss': rbm.losses['recon_loss']
    #     }
    #     # Also store the trained model itself for weight hist, error, updates
    #     rbm_models[ndim] = rbm

    # # -------------------------------------------------------------------
    # # 1) Plot reconstruction-loss convergence for both hidden sizes
    # # -------------------------------------------------------------------
    # plt.figure(figsize=(6,4))
    # for ndim, metrics in convergence.items():
    #     plt.plot(metrics['epochs'], metrics['recon_loss'], label=f"Hidden layer = {ndim}")
    # plt.xlabel("Epoch")
    # plt.ylabel("Reconstruction Loss")
    # plt.title("Convergence Comparison (Reconstruction Loss)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # -------------------------------------------------------------------
    # # 2) Plot weight histograms side-by-side
    # # -------------------------------------------------------------------
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # for i, ndim in enumerate(hidden_configs):
    #     rbm = rbm_models[ndim]
    #     axes[i].hist(rbm.weight_vh.flatten(), bins='auto', color='steelblue')
    #     axes[i].set_title(f"Weight Histogram (hidden={ndim})")
    #     axes[i].set_xlabel("Weight value")
    #     axes[i].set_ylabel("Frequency")
    # plt.tight_layout()
    # plt.show()

    # # -------------------------------------------------------------------
    # # 3) Plot reconstruction error (per batch) side-by-side
    # # -------------------------------------------------------------------
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # for i, ndim in enumerate(hidden_configs):
    #     rbm = rbm_models[ndim]
    #     axes[i].plot(range(len(rbm.errors)), rbm.errors, color='darkorange')
    #     axes[i].set_xlabel("Iteration")
    #     axes[i].set_ylabel("Error (||v0 - v1||)")
    #     axes[i].set_title(f"Error over iterations (hidden={ndim})")
    # plt.tight_layout()
    # plt.show()

    # # -------------------------------------------------------------------
    # # 4) Plot weight updates side-by-side
    # # -------------------------------------------------------------------
    # fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # for i, ndim in enumerate(hidden_configs):
    #     rbm = rbm_models[ndim]
    #     axes[i].plot(range(len(rbm.weight_updates)), rbm.weight_updates, color='green')
    #     axes[i].set_xlabel("Iteration")
    #     axes[i].set_ylabel("Mean Weight Update")
    #     axes[i].set_title(f"Weight Updates (hidden={ndim})")
    # plt.tight_layout()
    # plt.show()


    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=20)

    vis_hid = dbn.rbm_stack['vis--hid']
    hid_pen = dbn.rbm_stack['hid--pen']
    pen_top = dbn.rbm_stack['pen+lbl--top']

    # Print the reconstruction errror for each layer
    plt.figure()
    plt.plot(vis_hid.losses['epoch'], vis_hid.losses['recon_loss'], label="vis--hid")
    plt.plot(hid_pen.losses['epoch'], hid_pen.losses['recon_loss'], label="hid--pen")
    plt.plot(pen_top.losses['epoch'], pen_top.losses['recon_loss'], label="pen+lbl--top")
    plt.xlabel("Iteration")
    plt.ylabel("Reconstruction Loss")
    plt.title("Reconstruction Loss for each layer")
    plt.legend()
    plt.show()
 


    # labels_train = dbn.recognize(train_imgs, train_lbls)

    # plot_mean_label_evolution(labels_train)
    
    # labels_test = dbn.recognize(test_imgs, test_lbls)
    
    final_digits = []

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        vis = dbn.generate(digit_1hot, name="rbms")
        final_digits.append(vis)
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(final_digits[i], cmap='binary')
        ax.set_title(f"Digit {i}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


    ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")




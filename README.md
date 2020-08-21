# Variational Auto Encoder (VAE) task

This is the complete VAE task for the Deep Learning and Neural Networks course 2020.

Straight forward implementation of backpropagation for Autoencoder (given) and Variational Autoencoder.



## Training
- Training the model on the GreyFace dataset https://cs.nyu.edu/~roweis/data/frey_rawface.jpg with 1965. You can manually change the hyperparameters (the neural network size, learning rate, etc) to play around with the code a little bit.

```
python train_ae train
```

## Gradient Checking
- Checking the gradient correctness. This step is normally important when implementing back-propagation. The idea of grad-check is actually very simple:

+ We need to know how to verify the correctness of the back-prop implementation.
+ In order to do that we rely on comparison with the gradients computed using numerical differentiation
+ For each weight in the network we will have to do the forward pass twice (one by increasing the weight by \delta, and one by decreasing the weight by \delta)
+ The difference between two forward passes gives us the gradient for that weight
+ (maybe the code will be self-explanationable)

```
python train_ae.py gradcheck
```


## Evaluation / Reconstruction
- Using the model for reconstruction. There are around 1965 images in the dataset, so you can enter the image ID from 0 to 1900 to see the image (left) and its reconstruction (right).

```
python train_ae.py eval
```

## Sampling 
- Using the model to sample from a random code. The code will then be randomly generated from a normal distribution N(0, I). This will be then passed to the decoder to generate the image. However, with this model I expect very much to see a **darkspawn** instead of human faces. The VAE model if successfully implemented, will be able to help us generate human faces from samples of a known distribution. I also provided a model with 256 hidden units and 16 latent units trained after 120000 steps and batch size 64. 
 
 ```
python train_ae.py sample
```

# Reference
1. Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

# FAQs

1. RuntimeWarning: invalid value encountered in double_scalars rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)“ ... might happen. This can happen randomly when the denominator is 0 (or so small) and is not really a problem during gradcheck. 
2. The gradcheck might take too long. We should reduce the network size (hidden size and latent size) before doing gradcheck. There are 2 * (network size) + 1 forward passes to be ran in the check, so it is better to do on a small scale network.

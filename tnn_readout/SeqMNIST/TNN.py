import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.distributions.bernoulli import Bernoulli
from torchvision.datasets import MNIST
from tqdm import tqdm


class TNN:
  def __init__(self, w, n_neurons, threshold, timesteps):
    self.w = w
    self.n_neurons = n_neurons
    self.threshold = threshold
    self.timesteps = timesteps
  
  def forward(self, X):
    Yspikes = torch.zeros((self.timesteps, X.shape[1], self.n_neurons))  # output spike train
    for t in range(self.timesteps):
      X_t = X[t,:,:]
      Yspikes[t,:,:] = torch.mm(X_t, self.w)
    '''
    for t in range(timesteps):
      X_t = X[t,:,:,:] # take a splice across time
      Yspikes[t,:,:] = torch.mm(X_t, self.w)
    
    potentials = torch.sum(Yspikes, axis=1)
    Yspikes[Yspikes < threshold] = 0
    Yspikes[Yspikes >= threshold] = 1
    Ytimes = timesteps-torch.sum(Yspikes,axis=1)

    # apply k-WTA inhibition
    idx = np.lexsort((-potentials.numpy(),Ytimes.numpy()))
    loser_idx = idx[k:] # indexes of the num_neurons-k losers
    Ytimes[loser_idx]=timesteps
    Yspikes[loser_idx,:]=0
    return [Yspikes, Ytimes]'''
    return Yspikes

  def STDP(Xtimes, Ytimes, num_neurons, w, wmax, timesteps):

    # probabilities
    ucapture = Bernoulli(torch.tensor([10/128]))
    uminus = Bernoulli(torch.tensor([10/128]))
    usearch = Bernoulli(torch.tensor([1/128]))
    ubackoff = Bernoulli(torch.tensor([96/128]))
    umin = Bernoulli(torch.tensor([4/128]))

    new_w = -1*torch.ones(w.shape)

    for n in range(num_neurons):
        y = Ytimes[n]
        y = y.repeat(Xtimes.shape[0],Xtimes.shape[1],Xtimes.shape[2])
        w_n = w[:,:,:,n]

        fplus = Bernoulli((w_n/wmax)*(2-(w_n/wmax))).sample()
        fminus = Bernoulli((1-(w_n/wmax))*(1+(w_n/wmax))).sample()
        ucapture_sample = ucapture.sample(sample_shape=w_n.size()).squeeze()
        uminus_sample = uminus.sample(sample_shape=w_n.size()).squeeze()
        usearch_sample = usearch.sample(sample_shape=w_n.size()).squeeze()
        ubackoff_sample = ubackoff.sample(sample_shape=w_n.size()).squeeze()

        umin_sample1 = umin.sample(sample_shape=w_n.shape).squeeze()
        umin_sample2 = umin.sample(sample_shape=w_n.shape).squeeze()
        umin_sample3 = umin.sample(sample_shape=w_n.shape).squeeze()
        umin_sample4 = umin.sample(sample_shape=w_n.shape).squeeze()

        b1_cond_true =  w_n + ucapture_sample*torch.max(fplus,umin_sample1)
        b1 = torch.where((Xtimes < timesteps) & (y < timesteps) & (Xtimes <= y), b1_cond_true,w_n)

        b2_cond_true = b1-uminus_sample*torch.max(fminus,umin_sample2)
        b2 = torch.where((Xtimes < timesteps) & (y < timesteps) & (Xtimes > y), b2_cond_true,b1)

        b3_cond_true = b2 + usearch_sample*torch.max(fplus,umin_sample3)
        b3 = torch.where((Xtimes < timesteps) & (y == timesteps), b3_cond_true,b2)

        b4_cond_true = b3-ubackoff_sample*torch.max(fminus,umin_sample4)
        b4 = torch.where((Xtimes == timesteps) & (y < timesteps),b4_cond_true,b3)
        
        new_w[:,:,:,n] = b4.clone()

    # clamp weights within region
    new_w[new_w > wmax] = wmax
    new_w[new_w < 0] = 0
    return new_w

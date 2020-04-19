import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.uniform import Uniform
from torchvision.datasets import MNIST
from tqdm import tqdm


class TNN:
    def __init__(self, w_shape, n_neurons, threshold, timesteps):
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.timesteps = timesteps
        #self._init_w(w_shape)
        self.w = torch.zeros(w_shape).int()

    # initialize TNN weights using a uniform normal distribution
    def _init_w(self, shape):
        low = 0.0
        high = 2
        init_w_distr = Uniform(torch.tensor([low]), torch.tensor([high]))
        self.w = init_w_distr.sample(sample_shape=shape).squeeze().int()
        return

    def print_w(self):
        print(self.w)

    def forward(self, X):
        # output spike train
        Yspikes = torch.zeros((self.timesteps, X.shape[1], self.n_neurons))
        for t in range(self.timesteps):
            X_t = X[t, :, :].int()
            Yspikes[t, :, :] = torch.matmul(X_t, self.w)

        # threshold
        Yspikes[Yspikes < self.threshold] = 0
        Yspikes[Yspikes >= self.threshold] = 1

        Yspikes = self.wta_inhibibition(Yspikes)
        Ytimes = self.timesteps - Yspikes.sum(dim=0)
        return Ytimes

    def wta_inhibibition(self, Yspikes):
        Ytimes = Yspikes.sum(0)
        max_val = Ytimes.max(dim=1)[0] # get the spike time of the winner
        idx = (Ytimes==max_val).nonzero() # get indices of possible winners
        winner_idx = np.random.randint(0, len(idx)) # randomly choose a winner
        result = torch.zeros(Yspikes.shape)
        result[:, :, winner_idx] = Yspikes[:, :, winner_idx]
        return result

    def stdp(self, Xtimes, Ytimes):
        w_max = self.timesteps
        w_shape = self.w.shape
        self.w = self.w.float()  # cast the weight matrix as a float just for this function

        # probability parameters for the relevant distributions
        ucapture = Bernoulli(torch.tensor([10/128]))
        uminus = Bernoulli(torch.tensor([10/128]))
        usearch = Bernoulli(torch.tensor([1/128]))
        ubackoff = Bernoulli(torch.tensor([106/128]))
        umin = Bernoulli(torch.tensor([4/128]))

        # samples from the above probability distributions, each with the same shape as the w matrix
        ucapture_sample = ucapture.sample(sample_shape=w_shape).squeeze()
        uminus_sample = uminus.sample(sample_shape=w_shape).squeeze()
        usearch_sample = usearch.sample(sample_shape=w_shape).squeeze()
        ubackoff_sample = ubackoff.sample(sample_shape=w_shape).squeeze()

        # 4 samples from the umin distribution, with the same shape as the w matrix
        # we take 4 samples here for use in each of the 4 conditional branches of the STDP learning rule
        umin_sample1 = umin.sample(sample_shape=w_shape).squeeze()
        umin_sample2 = umin.sample(sample_shape=w_shape).squeeze()
        umin_sample3 = umin.sample(sample_shape=w_shape).squeeze()
        umin_sample4 = umin.sample(sample_shape=w_shape).squeeze()

        # here we define and sample from the F+ and F- distributions
        fplus = Bernoulli((self.w.float()/w_max) * (2-(self.w.float()/w_max))).sample()
        fminus = Bernoulli((1-(self.w.float()/w_max)) * (1+(self.w.float()/w_max))).sample()

        # reshape Xtimes and Ytimes so that they are the same shape (namely the shape of the weight matrix)
        # this is so that we can proceed with tensor processesing as opposed to iterating over the dimensions
        n_res_neurons = Xtimes.shape[1]
        Xtimes = Xtimes.transpose(0, 1).repeat(1, self.n_neurons)
        Ytimes = Ytimes.repeat(n_res_neurons, 1)

        # STDP LEARNING RULE (using tensor processing)
        b1_cond_true = self.w + ucapture_sample*torch.max(fplus, umin_sample1)
        self.w = torch.where((Xtimes < self.timesteps) & (Ytimes < self.timesteps) & (Xtimes <= Ytimes), b1_cond_true, self.w)

        b2_cond_true = self.w-uminus_sample*torch.max(fminus, umin_sample2)
        self.w = torch.where((Xtimes < self.timesteps) & (Ytimes < self.timesteps) & (Xtimes > Ytimes), b2_cond_true, self.w)

        b3_cond_true = self.w + usearch_sample*torch.max(fplus, umin_sample3)
        self.w = torch.where((Xtimes < self.timesteps) & (Ytimes == self.timesteps), b3_cond_true, self.w)

        b4_cond_true = self.w-ubackoff_sample*torch.max(fminus, umin_sample4)
        self.w = torch.where((Xtimes == self.timesteps) & (Ytimes < self.timesteps), b4_cond_true, self.w)

        # cast the weight matrix back to an int, and clamp the weights within [0,wmax]
        self.w = self.w.clamp(0, w_max).int()
        return

#!/usr/bin/env python
# coding: utf-8

# ##### YOU DON'T NEED TO CHANGE ANYTHING IN THIS CODE, EXCEPT FOR CHANGING ANY PARAMETERS IN [2] #######

# In[1]:


### Importing Libraries ###

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
from torchvision import transforms

from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils

import numpy as np
import matplotlib.pyplot as plt
import time


# In[2]:


### Importing MNIST Images and Preprocessing ###


### PARAMETERS THAT YOU CAN CHANGE FOR SECTION 8.3 ###

rf_size = 5 # Receptive field size that will be provided as input to the column
num_neurons = 16 # Number of excitatory neurons in the column
startposition = 13 # Start position of the receptive field w.r.t. top left corner of image
threshold = 32 # Firing threshold for every excitatory neuron
timesteps = 8 # Resolution for timesteps and weights

#####################################################


# Performs pre-processing on the input image
class PreProcTransform:
    def __init__(self, filter, timesteps = timesteps):
        self.to_tensor = transforms.ToTensor() # Convert to tensor
        self.filter = filter # Apply OnOff filtering
        self.temporal_transform = utils.Intensity2Latency(timesteps) # Convert pixel values to time
                                                    # Higher value corresponds to earlier spiketime
        self.crop = utils.Crop(startposition, rf_size) # Crop the image to form the receptive field
        
    def __call__(self, image):
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0) # Adds a temporal dimension at the beginning
        image = self.filter(image)
        temporal_image = self.temporal_transform(image)
        temporal_image = temporal_image.sign() # This will create spikes
        return self.crop(temporal_image)

kernels = [utils.OnKernel(3), utils.OffKernel(3)]
inchannels = len(kernels)

filter = utils.Filter(kernels, padding = 2, thresholds = 50)
preproc = PreProcTransform(filter)

# Defining iterator for loading train and test data
data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = preproc))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = preproc))
MNIST_trainLoader = DataLoader(MNIST_train, batch_size=1000, shuffle=True)
MNIST_testLoader = DataLoader(MNIST_test, batch_size=1000, shuffle=True)


# In[3]:


### Column Definition ###

class Column(nn.Module):
    def __init__(self, num_neurons, threshold):
        super(Column, self).__init__()
        self.k = num_neurons
        self.thresh = threshold
        # Local Convolution layer which creates columns with unique weights (NOT shared weights). The
        # number of columns is based on input_size, kernel_size and stride. Here since we are simulating
        # only one column, the input_size and kernel_size are kept same.
        self.ec = snn.LocalConvolution(input_size=rf_size,
                                       in_channels=inchannels,
                                       out_channels=self.k,
                                       kernel_size=rf_size,
                                       stride=1)
        # STDP module which implements the given STDP rule for the above layer (a single column in this case)
        self.stdp = snn.ModSTDP(self.ec, 10/128, 10/128, 1/128, 96/128, 4/128, maxweight = timesteps)
        
    def forward(self, rec_field):
        ### Start of Excitatory Column ###
        out = self.ec(rec_field)
        spike, pot = sf.fire(out, self.thresh, True)
        ### End of Excitatory Column ###
        ### Start of Lateral Inhibition ###
        out = sf.pointwise_inhibition(pot).sign()
        ### End of Lateral Inhibition ###
        return out


# In[4]:


### Column Initialization ###

MyColumn = Column(num_neurons, threshold)


# In[5]:


### Training a Column ###

for epochs in range(1):
    start = time.time()
    for data, target in MNIST_trainLoader:
        for i in range(len(data)):
            # Passing data through the column
            out = MyColumn(data[i])
            # STDP training for each input is performed here
            MyColumn.stdp(data[i],out)
    end = time.time()
    print("Training done under ", end-start)


# In[6]:


### Testing a Column and Computing Coverage and Purity Metrics ###

table    = torch.zeros((num_neurons,10))
pred     = torch.zeros(num_neurons)
totals   = torch.zeros(num_neurons)

count = 0
for data, target in MNIST_testLoader:
    for i in range(len(data)):
        count += 1
        out = MyColumn(data[i]).squeeze()
        out = torch.sum(out,0)
        temp = torch.nonzero(out)
        if temp.size(0) != 0:
            table[temp[0][0], target[i]] += 1
            
print("Confusion Matrix:")
print(table)

maxval = torch.max(table, 1)[0]
totals = torch.sum(table, 1)
pred = torch.sum(maxval)
covg_cnt = torch.sum(totals)

print("Purity: ", pred/covg_cnt)
print("Coverage: ", covg_cnt/count)


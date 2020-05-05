
import torch.nn as nn
import math
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms

from time import time as t
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import RankOrderEncoder
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input
from bindsnet.network.topology import Conv2dConnection, Connection
from bindsnet.utils import get_square_weights
from bindsnet.analysis.plotting import (
	plot_input,
	plot_spikes,
	plot_voltages,
	plot_weights,
	plot_conv2d_weights,
	plot_voltages,
)

from TNN import *

def seqMnistSimVanilla(examples, enum_dataloader, curr_i, network, time, spikes,
	train=True, plot=False, print_str=None, I_NAME="I", TNN_NAME="TNN_1", input_slice=28):
	if not train and print_str is None:
		print_str = "Testing"
	n_iters = examples
	test_pairs = []
	pbar = tqdm(enum_dataloader)
	for (i, dataPoint) in pbar:
		if i-curr_i > n_iters:
			break

		datum = dataPoint["encoded_image"].view(time, 1, 1, input_slice, input_slice)
		label = dataPoint["label"]
		pbar.set_description_str("%s progress: (%d / %d)" % (print_str, i-curr_i, n_iters))
		for row in range(input_slice):
			network.run(inputs={I_NAME: datum[:,:,:,row,:]}, time=time, input_time_dim=1)
		test_pairs.append([spikes[TNN_NAME].get("s").view(time, -1).sum(0), label])
		network.reset_state_variables()

	return test_pairs

# Splits input row between 2 input layers:
def seqMnistSimSplit(examples, enum_dataloader, curr_i, network, time, spikes,
	train=True, plot=False, print_str=None, slice_size=16, input_slice=28):
	if not train and print_str is None:
		print_str = "Testing"
	n_iters = examples
	test_pairs = []
	pbar = tqdm(enum_dataloader)
	for (i, dataPoint) in pbar:
		if i-curr_i > n_iters:
			break

		datum = dataPoint["encoded_image"].view(time, 1, 1, input_slice, input_slice)
		label = dataPoint["label"]
		pbar.set_description_str("%s progress: (%d / %d)" % (print_str, i-curr_i, n_iters))
		for row in range(input_slice):
			input_slices = {
				"I_a":datum[:,:,:,row,:slice_size],
				"I_b":datum[:,:,:,row,input_slice-slice_size:]
			}
			network.run(inputs=input_slices, time=time, input_time_dim=1)
		test_pairs.append([spikes["rTNN_1"].get("s").view(time, -1).sum(0), label])
		network.reset_state_variables()

	return test_pairs

# Concatenate rsult at end of each wave to training examples.
def seqMnistSimConcat(examples, enum_dataloader, curr_i, network, time, spikes,
	train=True, plot=False, print_str=None, I_NAME="I", TNN_NAME="TNN_1", input_slice=28):
	if not train and print_str is None:
		print_str = "Testing"
	n_iters = examples
	test_pairs = []
	pbar = tqdm(enum_dataloader)
	for (i, dataPoint) in pbar:
		if i-curr_i > n_iters:
			break
		example_data = []
		datum = dataPoint["encoded_image"].view(time, 1, 1, input_slice, input_slice)
		label = dataPoint["label"]
		pbar.set_description_str("%s progress: (%d / %d)" % (print_str, i-curr_i, n_iters))
		for row in range(input_slice):
			network.run(inputs={I_NAME: datum[:,:,:,row,:]}, time=time, input_time_dim=1)
			example_data.append(spikes[TNN_NAME].get("s").view(time, -1).sum(0))
		input_vec = torch.cat(example_data, 0).squeeze()
		test_pairs.append([input_vec, label])
		network.reset_state_variables()

	return test_pairs

# Define logistic regression model using PyTorch.
class LogReg(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogReg, self).__init__()
        # h = int(input_size/2)
        self.linear_1 = nn.Linear(input_size, num_classes)
        # self.linear_1 = nn.Linear(input_size, h)
        # self.linear_2 = nn.Linear(h, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        # out = torch.sigmoid(self.linear_2(out))
        return out


def train_readout(n_epochs, training_pairs, model, optimizer, criterion):
	# Training the Model
	print("\n Training the read out")
	pbar = tqdm(enumerate(range(n_epochs)))
	for epoch, _ in pbar:
	    avg_loss = 0
	    for i, (s, l) in enumerate(training_pairs):
	        # Forward + Backward + Optimize
	        optimizer.zero_grad()
	        outputs = model(s)
	        label = torch.zeros(1, 1, 10).float()
	        label[0, 0, l] = 1.0
	        loss = criterion(outputs.view(1, 1, -1), label)
	        avg_loss += loss.data
	        loss.backward()
	        optimizer.step()

	    pbar.set_description_str(
	        "Epoch: %d/%d, Loss: %.4f"
	        % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
	    )
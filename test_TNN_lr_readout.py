# Test for TNN column implementation
# Based off of conv_mnist example in bindnet
# Uses our temporal neuron node implementation with k-WTA.

import math
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
from time import time as t
from tqdm import tqdm
from bindsnet.datasets import MNIST
from bindsnet.encoding import RankOrderEncoder
from bindsnet.encoding import SingleEncoder
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
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils

# Command line parsing
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.set_defaults(plot=False, gpu=False, train=True)
args = parser.parse_args()
seed = args.seed
n_epochs = args.n_epochs
n_test = args.n_test
examples = args.examples
time = args.time
dt = args.dt
intensity = args.intensity
train = args.train
plot = args.plot
gpu = args.gpu
device_id =  0
gpu = False
if gpu and torch.cuda.is_available():
	torch.cuda.set_device(device_id)
	# torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
	torch.manual_seed(seed)

# Parameters for TNN
input_size = 28*28
tnn_layer_sz = 80
num_timesteps = 8
tnn_thresh = 250
max_weight = num_timesteps
num_winners = 3
time = num_timesteps

# TNN Network Build
network = Network(dt=1)
input_layer = Input(n=input_size)

tnn_layer_1a = TemporalNeurons( 
	n=tnn_layer_sz, 
	timesteps=num_timesteps, 
	threshold=tnn_thresh, 
	num_winners=num_winners
	)


T1a = Connection( 
	source=input_layer,
	target=tnn_layer_1a,
	w = 0.5 * max_weight * torch.rand(input_layer.n, tnn_layer_1a.n),
	update_rule=TNN_STDP,
	ucapture = 	10/128,
	uminus =	10/128,
	usearch = 	1/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = max_weight
	)

network.add_layer(input_layer, name="I")
network.add_layer(tnn_layer_1a, name="TNN_1a")
network.add_connection(T1a, source="I", target="TNN_1a") 


spikes = {}
for l in network.layers:
	spikes[l] = Monitor(network.layers[l], ["s"], time=num_timesteps)
	network.add_monitor(spikes[l], name="%s_spikes" % l)

dataset = MNIST(
	RampNoLeakTNNEncoder(time=num_timesteps, dt=1),
	None,
	root=os.path.join("..", "..", "data", "MNIST"),
	download=True,
	transform=transforms.Compose(
		[transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
	),
)

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
	dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=gpu
)

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None

n_iters = examples
training_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i >= n_iters:
		break
	print(i)
	datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)#.to(device_id)
	label = dataPoint["label"]
	pbar.set_description_str("Pre-train progress: (%d / %d)" % (i, n_iters))
	network.run(inputs={"I": datum}, time=time)
	training_pairs.append([spikes["TNN_1a"].get("s").sum(0), label])
	if plot:
		inpt_axes, inpt_ims = plot_input(
			dataPoint["image"].view(28, 28),
			datum.view(time, 784).sum(0).view(28, 28),
			label=label,
			axes=inpt_axes,
			ims=inpt_ims,
		)
		spike_ims, spike_axes = plot_spikes(
			{layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
			axes=spike_axes,
			ims=spike_ims,
		)
		weights_im = plot_weights(
			get_square_weights(T1a.w, int(math.ceil(math.sqrt(tnn_layer_sz))), 28), 
			im=weights_im, wmin=0, wmax=max_weight
		)

		plt.pause(1e-8)

	network.reset_state_variables()


# Define logistic regression model using PyTorch.
class NN(nn.Module):
	def __init__(self, input_size, num_classes):
		super(NN, self).__init__()
		# h = int(input_size/2)
		self.linear_1 = nn.Linear(input_size, num_classes)
		# self.linear_1 = nn.Linear(input_size, h)
		# self.linear_2 = nn.Linear(h, num_classes)

	def forward(self, x):
		out = torch.sigmoid(self.linear_1(x.float().view(-1)))
		# out = torch.sigmoid(self.linear_2(out))
		return out


# Create and train logistic regression model on reservoir outputs.
model = NN(tnn_layer_sz,10)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)


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

# Test

table = torch.zeros((tnn_layer_sz, 10))
pred = torch.zeros(tnn_layer_sz)
totals = torch.zeros(tnn_layer_sz)
count = 0
n_iters = examples
test_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i <= examples:
		continue
	if i >= n_iters + examples:
		break
	print(i)
	datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)
	label = dataPoint["label"]
	pbar.set_description_str("Test progress: (%d / %d)" % (i, n_iters))
	network.run(inputs={"I": datum}, time=time)
	test_pairs.append([spikes["TNN_1a"].get("s").sum(0), label])

	count += 1
	out = torch.sum(spikes["TNN_1a"].get("s").int().squeeze(), dim=0)
	temp = torch.nonzero(out)
	if temp.size(0) != 0:
		table[temp[0][0], label] += 1
	network.reset_state_variables()


# Test the Model
correct, total = 0, 0
for s, label in test_pairs:
	outputs = model(s)
	_, predicted = torch.max(outputs.data.unsqueeze(0), 1)
	total += 1
	correct += int(predicted == label.long())

print(
	"\n Accuracy of the model on %d test images: %.2f %%"
	% (n_iters, 100 * correct / total)
)


print("\n\n Confusion Matrix:")
print(table)

maxval = torch.max(table, 1)[0]
totals = torch.sum(table, 1)
pred = torch.sum(maxval)
covg_cnt = torch.sum(totals)

print("Purity: ", pred/covg_cnt)
print("Coverage: ", covg_cnt/count)

exit()














# training_pairs = []
pbar = tqdm(enumerate(dataloader))
n_iters= 1000
for (i, dataPoint) in pbar:
	if i > n_iters:
		break
	datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)#.to(device_id)
	label = dataPoint["label"]
	pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

	network.run(inputs={"I": datum}, time=time)
	# training_pairs.append([spikes["TNN_2"].get("s").int().squeeze(), label])

	if plot and i%100==0:

		inpt_axes, inpt_ims = plot_input(
			dataPoint["image"].view(28, 28),
			datum.view(time, 784).sum(0).view(28, 28),
			label=label,
			axes=inpt_axes,
			ims=inpt_ims,
		)
		spike_ims, spike_axes = plot_spikes(
			{layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
			axes=spike_axes,
			ims=spike_ims,
		)
		if (i == 0):
			for axis in spike_axes:
				axis.set_xticks(range(time))
				axis.set_xticklabels(range(time))	
			
			for l,a in zip(network.layers, spike_axes):
				a.set_yticks(range(network.layers[l].n))	

		weights_im = plot_weights(
			get_square_weights(T1a.w, int(math.ceil(math.sqrt(tnn_layer_sz))), 28), 
			im=weights_im, wmin=0, wmax=max_weight
		)
		weights_im2 = plot_weights(
			T2a.w, 
			im=weights_im2, wmin=0, wmax=l2_max_weight
		)


		plt.pause(1e-1)
	network.reset_state_variables()
print("\n")
print("Press enter to continue to evaluation")
print("\n")
input()


# TEST LOOP
table = torch.zeros((tnn_layer_2_sz, 10))
pred = torch.zeros(tnn_layer_2_sz)
totals = torch.zeros(tnn_layer_2_sz)
count = 0

pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i > n_iters:
		break
	datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)#.to(device_id)
	label = dataPoint["label"]
	pbar.set_description_str("Test progress: (%d / %d)" % (i, n_iters))

	network.run(inputs={"I": datum}, time=time)
	#training_pairs.append([spikes["TNN_1"].get("s").int().squeeze(), label])
	
	count += 1
	out = torch.sum(spikes["TNN_2"].get("s").int().squeeze(), dim=0)

	temp = torch.nonzero(out)
	
	if temp.size(0) != 0:
		table[temp[0][0], label] += 1

	network.reset_state_variables()

print("\n\n Confusion Matrix:")
print(table)

maxval = torch.max(table, 1)[0]
totals = torch.sum(table, 1)
pred = torch.sum(maxval)
covg_cnt = torch.sum(totals)

print("Purity: ", pred/covg_cnt)
print("Coverage: ", covg_cnt/count)
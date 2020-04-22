# Test for TNN column implementation
# Based off of conv_mnist example in bindnet
# Uses our temporal neuron node implementation with k-WTA.

import math
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
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
tnn_layer_sz = 25
tnn_layer_2_sz = 10
num_timesteps = 16
tnn_thresh = 256
tnn_thresh_2 = num_timesteps
max_weight = num_timesteps/2
l2_max_weight = max_weight*8
num_winners = 1
time = num_timesteps

# TNN Network Build
network = Network(dt=1)
input_layer = Input(n=input_size)
plot=True
tnn_layer_1a = TemporalNeurons(
	n=tnn_layer_sz,
	timesteps=num_timesteps,
	threshold=tnn_thresh,
	num_winners=num_winners
	)
tnn_layer_1b = TemporalNeurons(
	n=tnn_layer_sz,
	timesteps=num_timesteps,
	threshold=tnn_thresh,
	num_winners=num_winners
	)

tnn_layer_2 = TemporalNeurons(
	n=tnn_layer_2_sz,
	timesteps=num_timesteps,
	threshold=tnn_thresh_2,
	num_winners=1
	)

T1a = Connection(
	source=input_layer,
	target=tnn_layer_1a,
	w = 0.5 * max_weight * torch.rand(input_layer.n, tnn_layer_1a.n),
	update_rule=TNN_STDP,
	ucapture = 	20/128,
	uminus =	15/128,
	usearch = 	4/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = max_weight
	)
T1b = Connection(
	source=input_layer,
	target=tnn_layer_1b,
	w = 0.5 * max_weight * torch.rand(input_layer.n, tnn_layer_1b.n),
	update_rule=TNN_STDP,
	ucapture = 	20/128,
	uminus =	15/128,
	usearch = 	4/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = max_weight
	)

T2a = Connection(
	source=tnn_layer_1a,
	target=tnn_layer_2,
	w = 0.5 * max_weight * torch.rand(tnn_layer_1a.n, tnn_layer_2.n),
	update_rule=TNN_STDP,
	ucapture = 	10/128,
	uminus =	10/128,
	usearch = 	10/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = l2_max_weight
	)
T2b = Connection(
	source=tnn_layer_1b,
	target=tnn_layer_2,
	w = 0.5 * max_weight * torch.rand(tnn_layer_1b.n, tnn_layer_2.n),
	update_rule=TNN_STDP,
	ucapture = 	10/128,
	uminus =	10/128,
	usearch = 	10/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = l2_max_weight
	)

network.add_layer(input_layer, name="I")
network.add_layer(tnn_layer_1a, name="TNN_1a")
network.add_layer(tnn_layer_1b, name="TNN_1b")
network.add_layer(tnn_layer_2, name="TNN_2")
network.add_connection(T1a, source="I", target="TNN_1a")
network.add_connection(T1b, source="I", target="TNN_1b")
network.add_connection(T2a, source="TNN_1a", target="TNN_2")
network.add_connection(T2b, source="TNN_1b", target="TNN_2")


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

# Added by Kyle for On-Off Encoding
#dataset = MNIST(
#	root=os.path.join("..", "..", "data", "MNIST"),
#	download=True,
#	transform=preproc,
#)

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
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i > n_iters:
		break
	datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)#.to(device_id)
	label = dataPoint["label"]
	pbar.set_description_str("Pre-train progress: (%d / %d)" % (i, n_iters))

	network.run(inputs={"I": datum}, time=time)

	network.reset_state_variables()
plot=True
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
	# spikes[layer_name].get("s").view(time, -1).sum(0) 
    #output_history[sum_over_time]

	if plot and i%10==0:

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

n_iters = examples
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

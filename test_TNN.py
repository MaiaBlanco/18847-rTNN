# Test for TNN column implementation
# Based off of conv_mnist example in bindnet
# Uses our temporal neuron node implementation with k-WTA.

import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms

from time import time as t
from tqdm import tqdm

from bindsnet.datasets import MNIST
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

print()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--n_filters", type=int, default=25)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
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
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
examples = args.examples
padding = args.padding
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
device_id =  0#args.device_id

input_size = 28*28
tnn_layer_sz = 5
num_timesteps = 8
tnn_thresh = 32
max_weight = num_timesteps

time = num_timesteps
dt = num_timesteps
gpu = False

if gpu and torch.cuda.is_available():
    torch.cuda.set_device(device_id)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.manual_seed(seed)

# build network:
network = Network(dt=1)
input_layer = Input(n=input_size)

tnn_layer_1 = TemporalNeurons( \
	n=tnn_layer_sz, \
	timesteps=num_timesteps, \
	threshold=tnn_thresh, \
	num_winners=1\
	)

C1 = Connection( 
	source=input_layer,
	target=tnn_layer_1,
	w = 0.5 * torch.rand(input_layer.n, tnn_layer_1.n),
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
network.add_layer(tnn_layer_1, name="TNN_1")
network.add_connection(C1, source="I", target="TNN_1") 

spikes = {}
for l in network.layers:
	spikes[l] = Monitor(network.layers[l], ["s"], time=num_timesteps)
	network.add_monitor(spikes[l], name="%s_spikes" % l)

dataset = MNIST(
	SingleEncoder(time=num_timesteps, dt=1),
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
voltage_ims = None
voltage_axes = None


n_iters = examples
training_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i > n_iters:
		break
	datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)#.to(device_id)
	label = dataPoint["label"]
	pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

	network.run(inputs={"I": datum}, time=time)
	training_pairs.append([spikes["TNN_1"].get("s").int().squeeze(), label])

	if plot:

		inpt_axes, inpt_ims = plot_input(
			dataPoint["image"].view(28, 28),
			datum.view(time, 784).sum(0).view(28, 28),
			label=label,
			axes=inpt_axes,
			ims=inpt_ims,
		)
		spike_ims, spike_axes = plot_spikes(
			{layer: spikes[layer].get("s").view(-1, time) for layer in spikes},
			axes=spike_axes,
			ims=spike_ims,
		)
		weights_im = plot_weights(
			get_square_weights(C1.w, 23, 28), 
			im=weights_im, wmin=0, wmax=max_weight
		)

		plt.pause(1e-8)
	network.reset_state_variables()
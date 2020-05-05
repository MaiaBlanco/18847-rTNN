import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
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
from TNN_utils import *

print()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--examples", type=int, default=1000)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--device_id", type=int, default=0)

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
device_id =  0#args.device_id

input_size = 28
input_slice = 28
# tnn_layer_sz = 50
rtnn_layer_sz = 100
num_timesteps = 16
# tnn_thresh = 32
rtnn_thresh = 32
max_weight = 16
max_weight_rtnn = 32
# num_winners_tnn = 2 
num_winners_rtnn = rtnn_layer_sz//10

time = num_timesteps

torch.manual_seed(seed)

# build network:
network = Network(dt=1)
input_layer_a = Input(n=input_slice)

rtnn_layer_1 = TemporalNeurons( 
    n=rtnn_layer_sz, 
    timesteps=num_timesteps, 
    threshold=rtnn_thresh, 
    num_winners=num_winners_rtnn
    )

buffer_layer_1 = TemporalBufferNeurons(n=rtnn_layer_sz, timesteps=num_timesteps)
buffer_layer_2 = TemporalBufferNeurons(n=rtnn_layer_sz, timesteps=num_timesteps)

stdp_tnn_params = {
    "ucapture":  20/128,
    "uminus":    20/128,
    "usearch":   2/128,
    "ubackoff":  96/128,
    "umin":      4/128,
    "maxweight": max_weight
}
stdp_rtnn_params = {
    "ucapture":  30/128,
    "uminus":    30/128,
    "usearch":   10/128,
    "ubackoff":  96/128,
    "umin":      16/128,
    "maxweight": max_weight_rtnn
}

# Feed-forward connections
w_rand_l1 = 0.1 * max_weight * torch.rand(input_layer_a.n, rtnn_layer_1.n)
FF1a = Connection(source=input_layer_a, target=rtnn_layer_1,
	w = w_rand_l1, timesteps = num_timesteps,
    update_rule=TNN_STDP, **stdp_tnn_params)

# Recurrent connections
w_eye_rtnn = torch.diag(torch.ones(rtnn_layer_1.n))
rTNN_to_buf1 = Connection(source=rtnn_layer_1, target=buffer_layer_1,
	w = w_eye_rtnn, update_rule=None)
buf1_to_buf2 = Connection(source=buffer_layer_1, target=buffer_layer_2,
    w = w_eye_rtnn, update_rule=None)

# Force recurrent connectivity to be sparse, but strong.
w_recur = max_weight * torch.rand(rtnn_layer_1.n, rtnn_layer_1.n)
w_recur[torch.rand(rtnn_layer_1.n, rtnn_layer_1.n) < 0.90] = 0
buf1_to_rTNN = Connection(
	source=buffer_layer_1,
	target=rtnn_layer_1,
	w = w_recur,
    timesteps = num_timesteps,
    update_rule=None )
w_recur = max_weight * torch.rand(rtnn_layer_1.n, rtnn_layer_1.n)
w_recur[torch.rand(rtnn_layer_1.n, rtnn_layer_1.n) < 0.90] = 0
buf2_to_rTNN = Connection(
    source=buffer_layer_2,
    target=rtnn_layer_1,
    w = w_recur,
    timesteps = num_timesteps,
    update_rule=None )


# Add all nodes to network:
network.add_layer(input_layer_a, name="I_a")
network.add_layer(rtnn_layer_1, name="rTNN_1")
network.add_layer(buffer_layer_1, name="BUF_1")
network.add_layer(buffer_layer_2, name="BUF_2")

# Add connections to network:
# (feedforward)
network.add_connection(FF1a, source="I_a", target="rTNN_1")
# (Recurrences)
network.add_connection(rTNN_to_buf1, source="rTNN_1", target="BUF_1")
network.add_connection(buf1_to_buf2, source="BUF_1", target="BUF_2")
network.add_connection(buf1_to_rTNN, source="BUF_1", target="rTNN_1")
network.add_connection(buf2_to_rTNN, source="BUF_2", target="rTNN_1")

# End of network creation

# Monitors:
spikes = {}
for l in network.layers:
	spikes[l] = Monitor(network.layers[l], ["s"], time=num_timesteps)
	network.add_monitor(spikes[l], name="%s_spikes" % l)


# Data and initial encoding:
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
	dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False
)


inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
weights_im3 = None
weights_im4 = None

enum_dataloader = enumerate(dataloader)
i_offset = 0
# Start training synapses via STDP:
seqMnistSimVanilla(examples, enum_dataloader, i_offset, network, time, spikes,
    train=True, plot=False, print_str="Pre-Training", 
    I_NAME="I_a", TNN_NAME="rTNN_1", input_slice=input_slice)
i_offset += examples

if plot:
    input("Press enter to continue to plotting...")
    pbar = tqdm(enumerate(dataloader))
    n_iters = 10
    for (i, dataPoint) in pbar:
        if i > n_iters:
            break
        datum = dataPoint["encoded_image"].view(time, 1, 1, input_slice, input_slice)
        label = dataPoint["label"]
        pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
        for row in range(input_slice):
            inpt_axes, inpt_ims = plot_input(
                dataPoint["image"].view(input_slice, input_slice),
                datum.view(time, input_slice, input_slice).sum(0)[row,:].view(1,input_slice)*128,
                #datum[:,:,:,row,:].sum(0).view(1,28),
                label=label,
                axes=inpt_axes,
                ims=inpt_ims,
            )
            input_slices = {
                "I_a":datum[:,:,:,row,:input_slice],
            }
            network.run(inputs=input_slices, time=time, input_time_dim=1)
            spike_ims, spike_axes = plot_spikes(
                {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
                axes=spike_axes,
                ims=spike_ims,
                )
            plt.pause(1e-4)
        for axis in spike_axes:
            axis.set_xticks(range(time))
            axis.set_xticklabels(range(time))

        for l,a in zip(network.layers, spike_axes):
            a.set_yticks(range(network.layers[l].n))

        weights_im = plot_weights(
            FF1a.w,
            im=weights_im, wmin=0, wmax=max_weight
            )
        weights_im3 = plot_weights(
            buf1_to_rTNN.w,
            im=weights_im3, wmin=0, wmax=max_weight_rtnn
            )
        weights_im2 = plot_weights(
            buf2_to_rTNN.w,
            im=weights_im2, wmin=0, wmax=max_weight_rtnn
            )
        plt.pause(1e-12)
        input()

        network.reset_state_variables()

# Stop network from training further:
network.train(mode=False)

# Generate training pairs for log reg readout:
training_pairs = seqMnistSimVanilla(examples, enum_dataloader, i_offset, network, time, spikes,
    train=True, plot=False, print_str="Readout Training", 
    I_NAME="I_a", TNN_NAME="rTNN_1", input_slice=input_slice)
i_offset += examples


# Create and train logistic regression model on reservoir outputs.
model = LogReg(rtnn_layer_sz, 10)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
train_readout(n_epochs, training_pairs, model, optimizer, criterion)

# Generate testing pairs for log reg test:
test_pairs = seqMnistSimVanilla(examples, enum_dataloader, i_offset, network, time, spikes,
    train=False, plot=False, print_str="Readout Testing", 
    I_NAME="I_a", TNN_NAME="rTNN_1", input_slice=input_slice)
i_offset += examples

# Test the Model
correct, total = 0, 0
for s, label in test_pairs:
    outputs = model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    correct += int(predicted == label.long())

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (examples, 100 * correct / total)
)

# Drop into pdb in case we want to save the network or anything.
pdb.set_trace()

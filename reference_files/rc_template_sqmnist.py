# Kyle Buettner
# Neuromorphic Computer Architecture
# Reservoir Computing Example on MNIST
# This document serves as a template
# I edited the GPU code out (just CPU-based)

# Imports
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder, RankOrderEncoder
from bindsnet.network import Network
from bindsnet.network.nodes import Input
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights
from TNN import *

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=500)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.set_defaults(plot=True, gpu=True, train=True)
args = parser.parse_args()


seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
examples = args.examples
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
#plot = args.plot
plot = False
gpu = args.gpu
#device_id = args.device_id
device_id = 0                   # Set to CPU only

input_size = 28*1
tnn_layer_sz = 30
num_timesteps = 8
tnn_thresh = 12
max_weight = num_timesteps
time = num_timesteps
# Set up seeds for application
np.random.seed(seed)
torch.manual_seed(seed)

# Create reservoir computing network
# Described in BindsNet paper
network = Network(dt=dt)
inpt = Input(28, shape=(1, 1, 28))
network.add_layer(inpt, name="I")
output = TemporalNeurons( \
	n=tnn_layer_sz, \
	timesteps=num_timesteps, \
	threshold=tnn_thresh, \
	num_winners=1\
	)           # n_neurons 500
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n),
    ucapture=10/128,
	uminus =	10/128,
	usearch = 	1/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = max_weight)

C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n) ,
    ucapture=10/128,
	uminus =	10/128,
	usearch = 	1/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = max_weight)

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

# Set up monitors for spikes
spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

# Set up voltage monitors
# voltages = {"O": Monitor(network.layers["O"], ["v"], time=time)}
# network.add_monitor(voltages["O"], name="O_voltages")

# Get MNIST training images and labels.
dataset = MNIST(
    RankOrderEncoder(time=time, dt=dt),         # Originally was Poisson
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)
# Accuracy, Rank Order Coding: 58.28%

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
n_iters = examples
training_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    # print(datum.shape)
    # print(datum[:,:,:,0,:].shape)
    for row in range(28):
        #print('here')
        network.run(inputs={"I": datum[:,:,:,row,:]}, time=time, input_time_dim=1)
    training_pairs.append([spikes["O"].get("s").sum(0), label])

    print(spikes['O'].get("s").view(time,-1))
    #print(spikes['O'].get("s").view(time,-1).shape)
    if plot:

        inpt_axes, inpt_ims = plot_input(
            datum.view(time, 784).sum(0).view(28, 28),
            dataPoint["image"].view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time,-1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        # voltage_ims, voltage_axes = plot_voltages(
        #     {layer: voltages[layer].get("v").view(-1, time) for layer in voltages},
        #     ims=voltage_ims,
        #     axes=voltage_axes,
        # )
        weights_im = plot_weights(
            get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        )
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

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

n_iters = examples
test_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
    if i > n_iters:
        break

    datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    for row in range(28):
        network.run(inputs={"I": datum[:,:,:,row,:]}, time=time, input_time_dim=1)
    test_pairs.append([spikes["O"].get("s").sum(0), label])

    if plot:
        inpt_axes, inpt_ims = plot_input(
            datum.view(time, 784).sum(0).view(28, 28),
            dataPoint["image"].view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        # voltage_ims, voltage_axes = plot_voltages(
        #     {layer: voltages[layer].get("v").view(-1, 250) for layer in voltages},
        #     ims=voltage_ims,
        #     axes=voltage_axes,
        # )
        weights_im = plot_weights(
            get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
        )
        weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

        plt.pause(1e-8)
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


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

print()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=500)
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
tnn_layer_sz = 300
num_timesteps = 16
tnn_thresh = 32
max_weight = num_timesteps
num_winners = 1 #tnn_layer_sz

time = num_timesteps

torch.manual_seed(seed)

# build network:
network = Network(dt=1)
input_layer = Input(n=input_size)

tnn_layer_1 = TemporalNeurons( \
	n=tnn_layer_sz, \
	timesteps=num_timesteps, \
	threshold=tnn_thresh, \
	num_winners=num_winners\
	)

buffer_layer_1 = TemporalBufferNeurons(
	n = tnn_layer_sz,
	timesteps = num_timesteps,
	)

C1 = Connection(
	source=input_layer,
	target=tnn_layer_1,
	w = 0.01 * max_weight * torch.rand(input_layer.n, tnn_layer_1.n),
	update_rule=TNN_STDP,
	ucapture = 	10/128,
	uminus =	10/128,
	usearch = 	4/128,
	ubackoff = 	96/128,
	umin = 		4/128,
	timesteps = num_timesteps,
	maxweight = max_weight
	)

w = torch.diag(torch.ones(tnn_layer_1.n))

TNN_to_buf = Connection(
	source=tnn_layer_1,
	target=buffer_layer_1,
	w = w,
	update_rule=None
	)

buf_to_TNN = Connection(
	source=buffer_layer_1,
	target=tnn_layer_1,
	w = max_weight * torch.rand(tnn_layer_1.n, tnn_layer_1.n),
	update_rule=None, #TNN_STDP,
	ucapture = 	10/128,
	uminus =	10/128,
	usearch = 	8/128,
	ubackoff = 	96/128,
	umin = 		16/128,
	timesteps = num_timesteps,
	maxweight = max_weight
	)

network.add_layer(input_layer, name="I")
network.add_layer(tnn_layer_1, name="TNN_1")
network.add_layer(buffer_layer_1, name="BUF")
# network.add_connection(C2, source="TNN_1", target="TNN_1")
network.add_connection(buf_to_TNN, source="BUF", target="TNN_1")
network.add_connection(TNN_to_buf, source="TNN_1", target="BUF")
network.add_connection(C1, source="I", target="TNN_1")

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
	dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
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
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)
    label = dataPoint["label"]
    pbar.set_description_str("Pre-Train progress: (%d / %d)" % (i, n_iters))
    for row in range(28):
        network.run(inputs={"I": datum[:,:,:,row,:]}, time=time)

    network.reset_state_variables()

if plot:
    n_iters += 10
    for (i, dataPoint) in pbar:
        if i > n_iters:
            break
        datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)
        label = dataPoint["label"]
        pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
        for row in range(28):
            inpt_axes, inpt_ims = plot_input(
                dataPoint["image"].view(28, 28),
                datum.view(time, 28, 28).sum(0)[row,:].view(1,28)*1024,
                #datum[:,:,:,row,:].sum(0).view(1,28),
                label=label,
                axes=inpt_axes,
                ims=inpt_ims,
            )
            network.run(inputs={"I": datum[:,:,:,row,:]}, time=time)
            spike_ims, spike_axes = plot_spikes(
                {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
                axes=spike_axes,
                ims=spike_ims,
                )
            plt.pause(1e-1)
        for axis in spike_axes:
            axis.set_xticks(range(time))
            axis.set_xticklabels(range(time))

        for l,a in zip(network.layers, spike_axes):
            a.set_yticks(range(network.layers[l].n))

        weights_im = plot_weights(
            C1.w,
            im=weights_im, wmin=0, wmax=max_weight
            )
        weights_im2 = plot_weights(
            buf_to_TNN.w,
            im=weights_im2, wmin=0, wmax=max_weight
            )
        plt.pause(1e-12)

        network.reset_state_variables()

training_pairs = []
n_iters = examples
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28)
    label = dataPoint["label"]
    pbar.set_description_str("Generate Readout Training Pairs progress: (%d / %d)" % (i, n_iters))
    for row in range(28):
        network.run(inputs={"I": datum[:,:,:,row,:]}, time=time)
    training_pairs.append([spikes["TNN_1"].get("s").view(time, -1).sum(0), label])
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
    test_pairs.append([spikes["TNN_1"].get("s").view(time, -1).sum(0), label])
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

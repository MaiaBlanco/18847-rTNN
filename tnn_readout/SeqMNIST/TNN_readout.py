import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib
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
from bindsnet.encoding import *
from bindsnet.network import Network
from bindsnet.network.nodes import Input

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights
from TNN import TNN
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=32)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--rf_size", type=int, default=8)
parser.add_argument("--tnn_neurons", type=int, default=16)
parser.add_argument("--tnn_time", type=int, default=32)
parser.set_defaults(plot=True, gpu=False, train=True)

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
plot = args.plot
gpu = args.gpu
device_id = args.device_id
rf_size = args.rf_size
tnn_neurons = args.tnn_neurons
tnn_time = args.tnn_time

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
if gpu and torch.cuda.is_available():
    torch.cuda.set_device(device_id)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.manual_seed(seed)


network = Network(dt=dt)
inpt = Input(rf_size, shape=(1, 1, rf_size))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-62 + np.random.randn(n_neurons).astype(float))
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=0.5 *torch.randn(output.n, output.n))

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"O": Monitor(network.layers["O"], ["v"], time=time)}
network.add_monitor(voltages["O"], name="O_voltages")

# Directs network to GPU
if gpu:
    network.to("cuda")

# Get MNIST training images and labels.
# Load MNIST data.
# Transformations includes center cropping to rf_size
dataset = MNIST(
    SingleEncoder(time=time, dt=dt),
    #PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.CenterCrop(rf_size), transforms.ToTensor(
        ), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)


# Simulate the reservoir to generate training data pairs
n_iters = examples
train_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, rf_size, rf_size)
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    for row in range(rf_size):
        network.run(inputs={"I": datum[:,:,:,row,:]}, time=time, input_time_dim=1)
    train_pairs.append([spikes["O"].get("s").sum(0), label])
    network.reset_state_variables()


# instantiate tnn column
tnn_col = TNN(w_shape=(n_neurons, tnn_neurons), n_neurons=tnn_neurons, timesteps=tnn_time, threshold=16)
# train the tnn column
n_iters = examples
pbar = tqdm(enumerate(train_pairs))
for i, pair in pbar:
    #pdb.set_trace()
    if i > n_iters:
        break
    # reformatting the spike times as spike trains like in SpykeTorch
    # Note: Xspikes has a time dimension
    Xtimes = pair[0]
    #Xtimes = 8*(Xtimes - torch.min(Xtimes)) / (torch.max(Xtimes)-torch.min(Xtimes))  # scale features
    Xspikes = torch.zeros((time, Xtimes.shape[0], Xtimes.shape[1]))
    for j in range(n_neurons):
        xtime = Xtimes[0, j]
        Xspikes[xtime:, 0, j] = 1
    pbar.set_description_str("Train TNN Progress: (%d / %d)" % (i, len(train_pairs)-1))
    # get output of tnn column
    Ytimes = tnn_col.forward(Xspikes)

    # do stdp
    tnn_col.stdp(Xtimes, Ytimes)

# Simulate the reservoir to generate testing data pairs
n_iters = examples
test_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, rf_size, rf_size)
    label = dataPoint["label"]
    pbar.set_description_str("Test progress: (%d / %d)" % (i, n_iters))
    for row in range(rf_size):
        network.run(inputs={"I": datum[:,:,:,row,:]}, time=time, input_time_dim=1)
    test_pairs.append([spikes["O"].get("s").sum(0), label])
    network.reset_state_variables()


# test the tnn column
table = torch.zeros((tnn_neurons, 10))
pred = torch.zeros(tnn_neurons)
totals = torch.zeros(tnn_neurons)

count = 0
n_iters = examples
pbar = tqdm(enumerate(test_pairs))
for i, pair in pbar:
    count += 1
    if i > n_iters:
        break
    # reformatting the spike times as spike trains like in SpykeTorch
    # Note: Xspikes has a time dimension
    Xtimes = pair[0]
    #Xtimes = 8*(Xtimes - torch.min(Xtimes)) / (torch.max(Xtimes)-torch.min(Xtimes))  # scale features
    Xspikes = torch.zeros((time, Xtimes.shape[0], Xtimes.shape[1]))
    for j in range(n_neurons):
        xtime = Xtimes[0, j]
        Xspikes[xtime:, 0, j] = 1
    
    pbar.set_description_str("Test TNN Progress: (%d / %d)" % (i, len(test_pairs)-1))
    # get output of tnn column
    Ytimes = tnn_col.forward(Xspikes)
    out = Ytimes.sum(dim=0)
    temp = torch.nonzero(tnn_time - out)
    if temp.size(0) != 0:
        table[temp[0][0], pair[1]] += 1

print("\n\n Confusion Matrix:")
print(table)

maxval = torch.max(table, 1)[0]
totals = torch.sum(table, 1)
pred = torch.sum(maxval)
covg_cnt = torch.sum(totals)

print("Purity: ", pred/covg_cnt)
print("Coverage: ", covg_cnt/count)

tnn_col.print_w()
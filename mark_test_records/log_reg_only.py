
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

input_size = 28*28
num_timesteps = 16
time = num_timesteps

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

training_pairs = []
test_pairs = []
pbar = enumerate(dataloader)
n_iters = 10
for (i, dataPoint) in pbar:
    datum = dataPoint["encoded_image"].view(time, 1, 1, 28*28)
    label = dataPoint["label"]
    if i <= n_iters:
        training_pairs.append([datum.sum(0).squeeze(), label])
    if i > n_iters and i < n_iters*2:
        test_pairs.append([datum.sum(0).squeeze(), label])


n_epochs = 10
# Create and train logistic regression model on direct outputs.
model = LogReg(28*28, 10)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
train_readout(n_epochs, training_pairs, model, optimizer, criterion)

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

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


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=500)    # 500
parser.add_argument("--n_epochs", type=int, default=200)    #100
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=28)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
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

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
if gpu and torch.cuda.is_available():
    torch.cuda.set_device(device_id)
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.manual_seed(seed)

network = Network(dt=dt)
inpt = Input(28, shape=(1, 1, 28))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-52 + np.random.randn(n_neurons).astype(float))
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=0.5 * torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))

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
dataset = MNIST(
    #RankOrderEncoder(time=time, dt=dt),
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

#dataset.data = torch.flatten(dataset.train_data, start_dim=1)
#dataset.data = dataset.data.unsqueeze(2).unsqueeze(3)
#print(dataset.data.shape)

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
    if i >= n_iters:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, 28)
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
    network.run(inputs={"I": datum}, time=time, input_time_dim=1)
    training_pairs.append([spikes["O"].get("s").sum(0), label])
    print(spikes["O"].get("s"))
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
model = NN(n_neurons, 10)
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
    if i <= examples:
        continue
    if i >= n_iters + examples:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, 28)
    label = dataPoint["label"]
    pbar.set_description_str("Test progress: (%d / %d)" % (i, n_iters))
    network.run(inputs={"I": datum}, time=time, input_time_dim=1)
    test_pairs.append([spikes["O"].get("s").sum(0), label])
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
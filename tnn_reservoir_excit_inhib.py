
import torch.nn as nn
import math
import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
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
parser.add_argument("--n_epochs", type=int, default=1000)
parser.add_argument("--n_test", type=int, default=1000)
parser.add_argument("--examples", type=int, default=5000)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--rf_size", type=int, default=28)


args = parser.parse_args()
rf_size = args.rf_size
seed = args.seed
n_epochs = args.n_epochs
n_test = args.n_test
examples = args.examples
time = args.time
dt = args.dt
intensity = args.intensity
train = args.train
plot = args.plot
gpu = True

input_layer_sz = rf_size
tnn_layer_sz = 512
num_timesteps = 8
tnn_thresh = 8
max_weight = num_timesteps
num_winners = 128
buf_sz = 1

time = num_timesteps
torch.manual_seed(seed)



# build network:
network = Network(dt=1)
input_layer = Input(n=input_layer_sz)

tnn_layer_1 = TemporalNeurons( \
	n=tnn_layer_sz, \
	timesteps=num_timesteps, \
	threshold=tnn_thresh, \
	num_winners=num_winners\
	)

tnn_layer_2 = TemporalNeurons( \
	n= int(tnn_layer_sz/2), \
	timesteps=num_timesteps, \
	threshold=tnn_thresh, \
	num_winners=num_winners\
	)


C1 = Connection(
	source=input_layer,
	target=tnn_layer_1,
	w = max_weight*torch.rand(input_layer.n, tnn_layer_1.n),
	update_rule=None,
	)

# C2 = Connection(
# 	source=input_layer,
# 	target=tnn_layer_2,
# 	w = max_weight*torch.rand(input_layer.n, tnn_layer_2.n),
# 	update_rule=None,
# 	)

C3 = Connection(
	source=tnn_layer_1,
	target=tnn_layer_2,
	w = max_weight*torch.rand(tnn_layer_1.n, tnn_layer_2.n),
	update_rule=None,
	)

C4 = Connection(
	source=tnn_layer_2,
	target=tnn_layer_1,
	w = -max_weight*torch.rand(tnn_layer_2.n, tnn_layer_1.n),
	update_rule=None,
	)



network.add_layer(input_layer, name="I")
network.add_layer(tnn_layer_1, name="TNN_1")
network.add_layer(tnn_layer_2, name="TNN_2")
network.add_connection(C1, source="I", target="TNN_1")
#network.add_connection(C2, source="I", target="TNN_2")
network.add_connection(C3, source="TNN_1", target="TNN_2")
network.add_connection(C4, source="TNN_2", target="TNN_1")

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
		[transforms.CenterCrop(rf_size), transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
	),
)


# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
	dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
)

training_pairs = []
n_iters = examples
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i > n_iters:
		break
	datum = dataPoint["encoded_image"].view(time, 1, rf_size, rf_size)
	label = dataPoint["label"]
	pbar.set_description_str("Generate Readout Training Pairs progress: (%d / %d)" % (i, n_iters))
	
	for row in range(rf_size):
		datum_r = datum[:,:,row,:]
		network.run(inputs={"I": datum_r}, time=time)

	tnn1_spikes = spikes["TNN_1"].get("s").view(time, -1).sum(0)
	training_pairs.append([tnn1_spikes.squeeze(), label])
	network.reset_state_variables()

# print(training_pairs)
# input()

# Define logistic regression model using PyTorch.
class LogisticRegression(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)    
		
	def forward(self, x):
		out = fn.softmax(self.linear(x))
		return out

# Create and train logistic regression model on reservoir outputs.

model = LogisticRegression(tnn_layer_sz,10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
logistic_regr_loader = torch.utils.data.DataLoader(dataset=training_pairs, batch_size=32, shuffle=True)
# Training the Model
print("\n Training the read out")
pbar = tqdm(enumerate(range(n_epochs)))
for epoch, _ in pbar:
	avg_loss = 0
	for i, (s, l) in enumerate(logistic_regr_loader):
		optimizer.zero_grad()
		outputs = model(s)
		loss = criterion(outputs, l.squeeze().long())
		avg_loss += loss.data
		loss.backward()
		optimizer.step()

	pbar.set_description_str(
		"Epoch: %d/%d, Loss: %.4f"
		% (epoch + 1, n_epochs, avg_loss / len(training_pairs))
	)


n_iters = n_test
test_pairs = []
pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i > n_iters:
		break
	datum = dataPoint["encoded_image"].view(time, 1, rf_size, rf_size)
	label = dataPoint["label"]
	pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))
	
	for row in range(rf_size):
		datum_r = datum[:,:,row,:]
		network.run(inputs={"I": datum_r}, time=time)

	tnn1_spikes = spikes["TNN_1"].get("s").view(time, -1).sum(0)
	test_pairs.append([tnn1_spikes.squeeze(), label])
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

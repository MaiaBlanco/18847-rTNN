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
on_off_size = input_size * 2
tnn_layer_sz = 10
num_timesteps = 8 # 64
tnn_thresh = 32
max_weight = num_timesteps
num_winners = 4 
time = num_timesteps

# SpykeTorch specific parameters for on-off encoding
rf_size = 28 # Receptive field size that will be provided as input to the column
startposition = 0 # Start position of the receptive field w.r.t. top left corner of image

# On-Off Transform class for Image
class PreProcTransform:
    def __init__(self, filter, timesteps=num_timesteps):
        self.to_tensor = transforms.ToTensor() # Convert to tensor
        self.filter = filter # Apply OnOff filtering
        self.temporal_transform = utils.Intensity2Latency(timesteps) # Convert pixel values to time
                                                    # Higher value corresponds to earlier spiketime
        self.crop = utils.Crop(startposition, rf_size) # Crop the image to form the receptive field
        
    def __call__(self, image):
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0) # Adds a temporal dimension at the beginning
        image = self.filter(image)
        temporal_image = self.temporal_transform(image)
        temporal_image = temporal_image.sign() # This will create spikes
        return self.crop(temporal_image)

kernels = [utils.OnKernel(3), utils.OffKernel(3)]
filter = utils.Filter(kernels, padding = 2, thresholds = 50)
preproc = PreProcTransform(filter)

# TNN Network Build
network = Network(dt=1)
input_layer = Input(n=on_off_size)


tnn_layer_1 = TemporalNeurons( \
	n=tnn_layer_sz, \
	timesteps=num_timesteps, \
	threshold=tnn_thresh, \
	num_winners=num_winners\
	)

C1 = Connection( 
	source=input_layer,
	target=tnn_layer_1,
	w = 0.5 * max_weight * torch.rand(input_layer.n, tnn_layer_1.n),
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

#dataset = MNIST(
	#PoissonEncoder(time=num_timesteps, dt=1),
	#None,
	#root=os.path.join("..", "..", "data", "MNIST"),
	#download=True,
	#transform=transforms.Compose(
	#	[transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
	#),
#)

# Added by Kyle for On-Off Encoding
dataset = MNIST(
	root=os.path.join("..", "..", "data", "MNIST"),
	download=True,
	transform=preproc,
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
	datum = dataPoint["encoded_image"]
	print(datum.shape)
	on_image = dataPoint["encoded_image"][:,:,0,:,:]
	on_image = torch.squeeze(on_image)
	on_image = torch.unsqueeze(on_image, 1)
	on_image = on_image.view(time, 1, input_size)
	off_image = dataPoint["encoded_image"][:,:,1,:,:]
	off_image = torch.squeeze(off_image)
	off_image = torch.unsqueeze(off_image, 1)
	off_image = off_image.view(time, 1, input_size)
	on_off_image = torch.cat((on_image, on_image), 2)
	label = dataPoint["label"]
	pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))
	print(on_off_image.shape)
	network.run(inputs={"I": on_off_image}, time=time)
	training_pairs.append([spikes["TNN_1"].get("s").int().squeeze(), label])

	if plot:
		spike_ims, spike_axes = plot_spikes(
			{layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
			axes=spike_axes,
			ims=spike_ims,
		)
		weights_im = plot_weights(C1.w,
			#get_square_weights(C1.w, int(math.ceil(math.sqrt(tnn_layer_sz))), 1568), 
			im=weights_im, wmin=0, wmax=max_weight
		)

		plt.pause(1e-8)
	network.reset_state_variables()

# TEST LOOP
table = torch.zeros((tnn_layer_sz, 10))
pred = torch.zeros(tnn_layer_sz)
totals = torch.zeros(tnn_layer_sz)
count = 0

pbar = tqdm(enumerate(dataloader))
for (i, dataPoint) in pbar:
	if i > n_iters:
		break
	datum = dataPoint["encoded_image"]
	on_image = dataPoint["encoded_image"][:,:,0,:,:]
	on_image = torch.squeeze(on_image)
	on_image = torch.unsqueeze(on_image, 1)
	on_image = on_image.view(time, 1, input_size)
	off_image = dataPoint["encoded_image"][:,:,1,:,:]
	off_image = torch.squeeze(off_image)
	off_image = torch.unsqueeze(off_image, 1)
	off_image = off_image.view(time, 1, input_size)
	on_off_image = torch.cat((on_image, on_image), 2)
	label = dataPoint["label"]
	pbar.set_description_str("Test progress: (%d / %d)" % (i, n_iters))

	network.run(inputs={"I": on_off_image}, time=time)
	#training_pairs.append([spikes["TNN_1"].get("s").int().squeeze(), label])
	
	count += 1
	out = torch.sum(spikes["TNN_1"].get("s").int().squeeze(), dim=0)

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
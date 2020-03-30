import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

# Get base class to implement our own neuron type
from bindsnet.network.nodes import Nodes


class TemporalNeuronNodes(Nodes):
	"""
	Neuron sub-class based on Nodes base class in bindsNet

	This neuron implements the same functionality as we had in lab 1:
	temporal neurons that have a time resolution (number of steps),
	firing threshold (integer value), current value (integer),

	Methods are:
	* initialization to set the neuron parameters
	* forward method to take TEMPORAL inputs in the input receptive field and
	output a temporally-encoded response.
	* reset method to set the neuron back to normal operation after being inhibited
	or at the end of computation wave (gamma oscillation)
	* training function to update the weights (via STDP)
	* required _compute_decays() method, which won't do anything here.
	"""

	def __init__(
		self, 
		n: Optional[int] = None,
		shape: Optional[Iterable[int]] = None,
		traces: bool = False,
		traces_additive: bool = False,
		tc_trace: Union[float, torch.Tensor] = 20.0,
		trace_scale: Union[float, torch.Tensor] = 10.0,
		sum_input: bool = True,
		thresh: Union[int, torch.Tensor] = 8,
		**kwargs,
		) -> None:

		super().__init__(
			n=n,
			shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,	
		)

		# Init neuron weights here (random)


		# Other neuron setup goes here

	# Forward function needs to sum across the time domain:
	def forward(self, x_temporal) -> None:
		# self.v - voltages
		# self.thresh - firing threshold
		# self.s - self output (temporal coded)
		# Where are the weights? These are presynaptic. The x_temporal we 
		# receive here is post synaptic, and already scaled by synaptic weight.
		# Therefore each neuron just needs to integrate over time and threshold.
		
			
		# ...
		super().forward(x_temporal)

	def reset_state_variables(self) -> None:
		super.reset_state_variables()

		

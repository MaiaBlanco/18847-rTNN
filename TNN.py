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
from bindsnet.learning import LearningRule

class TemporalNeurons(Nodes):
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
		timesteps: int,
		threshold: Optional[int] = None,
		num_winners: Optional[int] = None,
		# Boilerplate inputs that I don't care about:
		traces: bool = False,
		traces_additive: bool = False,
		tc_trace: Union[float, torch.Tensor] = 20.0,
		trace_scale: Union[float, torch.Tensor] = 10.0,
		sum_input: bool = False, 
		thresh: Union[int, torch.Tensor] = 8,
		**kwargs,
		) -> None:

		super().__init__(
			n=n,
			shape=shape,
            traces=False, 
            traces_additive=False,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=False,	
		)

		# NOTE: neuron weights are initialized in the Connection class, 
		# which then handles pre- to post-synaptic scaling of inputs.
		# Set number of timesteps (this is also the threshold if otherwise not set)
		self.timesteps = timesteps
		if threshold is None:
			self.threshold = timesteps
		else:
			self.threshold = threshold
		self.register_buffer("thresh", torch.tensor(self.threshold), dtype=torch.int)
		# For storing summed inputs along time per neuron:
		self.register_buffer("cumulative_inputs", torch.IntTensor())
	
		if num_winners is None:
			self.inhibition = False
			self.num_winners = self.n
		else:
			self.inhibition = True
			self.num_winners = num_winners	
		# There already is a bytetensor in the base class called s for spikes
		# but having a buffer for output sums is still helpful if applying inhibition:
		self.register_buffer("output_sums", torch.ByteTensor())

	# Forward function needs to sum across the time domain:
	# x_temporal is a tensor of post-synaptic activations.
	# This means that each single spike (if present) along the time 
	# dimension has already been scaled by the weight for that 
	# input -> neuron connection, and then reduced into a vector.
	# Since this class handles multiple neurons, the input tensor
	# has dimensionality num_neurons x time.
	def forward(self, x_temporal) -> None:
		# self.v - voltages
		# self.thresh - firing threshold
		# self.s - self output (temporal coded)
		# Where are the weights? These are presynaptic. The x_temporal we 
		# receive here is post synaptic, and already scaled by synaptic weight.
		# Therefore each neuron just needs to integrate over time and threshold.
		last_dim = x_temporal.dim()-1
		self.cumulative_inputs = torch.cumsum(x_temporal, last_dim)
		self.s[self.cumulative_inputs >= self.threshold] = 1			
		self.pointwise_inhibition() # Apply inhibition to self.s
		# ...
		super().forward(x_temporal)

	def reset_state_variables(self) -> None:
		self.cumulative_inputs.zero()
		self.s.zero()
		super.reset_state_variables()

	# Apply pointwise inhibition to computed spike outputs
	def pointwise_inhibition(self) -> None:
		if not self.inhibition or self.num_winners >= self.n:
			return
		self.output_sums = torch.reshape(torch.sum(self.s, self.s.dim()-1), \
			(self.n))
		flattened_spikes = torch.reshape(torch.s, (self.n, self.timesteps))
		# First to fire will have higher output sum:
		indices = torch.argsort(self.output_sums, descending=True)
		# Use indices to clear neuron outputs from 
		# num_winners to n:
		losing_indices = indices[self.num_winners:]
		flattened_spikes[losing_indices,:] = 0
		self.s = torch.reshape(flattened_spikes, (*self.shape, self.timesteps)) 


		
class TNN_STDP(LearningRule):
    # language=rst
    """
    Learning rule for TNN neurons. This rule implements 
    spike-timing dependent plasticity (STDP) on the weights 
    held in a connection object.
    The weights are modified based on the presynaptic spikes 
    (held by the connection object) and the output spikes 
    (output by the TNN nodes object) AFTER WTA is applied.
   	Based on the causal relationship of an input and output spike, 
   	the weights are increased in strength or decreased.
    """

    def __init__(
        self,
        connection: AbstractConnection, # Basically the layer of nodes
        ucapture: float,
        uminus: float,
        usearch: float,
        ubackoff: float,
        umin: float,
        umax: float,
       	maxweight,: float 
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection,
            nu=None, # Outdated.
            reduction=None,
            weight_decay=0.0,
            **kwargs
        )

        self.connection = connection
        self.ucapture = ucapture
        self.uminus = uminus
        self.usearch = usearch
        self.ubackoff = ubackoff
        self.umin = umin
        self.maxweight = maxweight
        trand.manual_seed(0)        # set seed for determinism



    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        input_spikes = self.connection.source.s
        output_spikes = self.connection.target.s
        weights = self.connection.w.view()

        # Copied from lab 1 in spyketorch:
        # Find when input spike occurred (if at all) for each input channel 
        # and receptive field row and column:
        x_times = int(self.maxweight) - torch.sum(input_spikes.int(), input_spikes.dim()-1)
        # ^ should have shape connection.source.shape

        # Do the same for outputs (max weight = number of time steps):
        y_times = int(self.maxweight) - torch.sum(output_spikes.int(), output_spikes.dim()-1)
        # ^ should have shape connection.target.shape

        # Get tensor for umin
        umin_tensor = torch.full(weights.size(), self.umin, dtype=torch.float)
        
        # Get the input receptive field (here it's the whole thing. No slicing.)
        x_slice = x_times
        y_slice = y_times
        # Each output corresponds to one of connection.target.n neurons in the receptive field.
        # Each weight maps one temporal pixel in the receptive field (connection.source) to a neuron
        # in connection.target
        # So, we need to broadcast each input spike time in the receptive field to each 
        # neuron and vice-versa.
        desired_size = (y_times.size()[0], *x_times.size())
        bcast_y = torch.zeros(desired_size)
        bcast_x = torch.zeros(desired_size) 
        for i in range(self.layer.out_channels):
        	bcast_y[i,:,:,:] = y_slice[i]
        	bcast_x[i,:,:,:] = x_slice

        # Conditions for weight updates:
        A = bcast_x == self.maxweight
        B = bcast_y == self.maxweight
        C = bcast_x > bcast_y

        # The 5 cases:
        # 1. !A ^ !B ^ !C       increase with P=capture
        # 2. !A ^ !B ^ C        decrease with P=minus
        # 3. !A ^ B             increase with P=search
        # 4. A ^ !B             decrease weight with P=backoff
        # 5. A ^ B              No change

        # Get weights patch:
        weights_patch = torch.clone(self.layer.weight[neural_row,neural_col,:,:,:,:]).float()
        
        # Need 2 sets of probabilities for increment and decrement:
        probs_plus = torch.zeros_like(weights_patch).float()
        probs_minus = torch.zeros_like(weights_patch).float()
        probs_plus[~A & ~B & ~C] = self.ucapture
        probs_minus[~A & ~B & C] = self.uminus
        probs_plus[~A & B] = self.usearch
        probs_minus[A & ~B] = self.ubackoff
        # Implicitly all other entries are zero, which means no update will be applied.

        # Generate probabilities that weight updates occur:
        bernoulli_frame_plus  = torch.bernoulli(probs_plus)
        bernoulli_frame_minus = torch.bernoulli(probs_minus)

        # Division costs a lot more than multiply, so compute the inverse once:
        inv_max_weight = 1/self.maxweight
        F_probs_ratio = torch.mul(weights_patch, inv_max_weight)

        # Compute F +/- probabilities 
        F_minus_probs = (1-F_probs_ratio) * (1+F_probs_ratio)
        F_minus = torch.bernoulli(F_minus_probs)
        F_plus_probs = F_probs_ratio * (2 - F_probs_ratio)
        F_plus = torch.bernoulli(F_plus_probs)

        # add umin probability to F+/- probability:
        umin_bernoulli = torch.bernoulli(umin_tensor)
        F_plus = torch.max(F_plus, umin_bernoulli)
        F_minus = torch.max(F_minus, umin_bernoulli)

        # Apply updates to weights patch (ewise add)
        weights_patch = torch.add(weights_patch, bernoulli_frame_plus * F_plus)
        weights_patch = torch.add(weights_patch, -1 * bernoulli_frame_minus * F_minus)

        # Clamp outputs to range
        torch.clamp_(weights_patch, 0, self.maxweight)

        # Assign updated weights back to layer
        self.layer.weight[neural_row,neural_col,:,:,:,:] = weights_patch.int()

        super().update()
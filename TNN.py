import pdb
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
import torch.nn.functional as fn
from tqdm import tqdm

from time import time as t

# Get base class to implement our own neuron type
from bindsnet.network.nodes import Nodes
from bindsnet.network.topology import AbstractConnection, Connection
from bindsnet.network.monitors import AbstractMonitor
from bindsnet.learning.reward import AbstractReward
from bindsnet.network.network import Network
from bindsnet.learning import LearningRule
from bindsnet.encoding import Encoder

from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Dict, Iterable, Optional, Union, Sequence

class TemporalBufferNeurons(Nodes):
    """
    Neuron sub-class based on Nodes base class in bindsNet

    This neuron type is meant to enable synchronous-recurrent connections between 
    TemporalNeurons (defined below).
    Methods are:
    * initialization to set the neuron parameters
    * forward method to take in and buffer TEMPORAL inputs in the input,
     and output the temporally-encoded spikes received at the same time step 
     in the previous wave of computation.
    """

    def __init__(
        self, 
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        timesteps: int = 10,
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
    
        # There already is a bytetensor in the base class called s for spikes
        # but here we need buffers for inputs and outputs over time 
        #self.register_buffer("buffer1", 
        #    torch.zeros((self.timesteps, *self.shape)))#, dtype=torch.int8))
        #self.register_buffer("buffer2", 
        #    torch.zeros((self.timesteps, *self.shape)))#, dtype=torch.int8))
        self.buffer1 = torch.zeros((self.timesteps, *self.shape), dtype=torch.int8)
        self.buffer2 = torch.zeros((self.timesteps, *self.shape), dtype=torch.int8)
        
        self.counter = 0 # To keep track of time

    def forward(self, x) -> None:
        self.s[0,:] = self.buffer1[self.counter,:]
        self.buffer2[self.counter,:] = x #.clone()
        self.counter += 1
        if (self.counter == self.timesteps):
            self.wave_reset()
        super().forward(x)

    def wave_reset(self):
        self.counter = 0
        self.buffer1 = self.buffer2
    
    def reset_state_variables(self) -> None:
        self.buffer2.zero_()
        self.wave_reset()
        super().reset_state_variables()

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
        timesteps: int = 10,
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
        self.register_buffer("thresh", torch.tensor(self.threshold, dtype=torch.int))
        # For storing summed inputs along time per neuron:
        self.register_buffer("cumulative_inputs", torch.zeros(self.shape, dtype=torch.float))
    
        if num_winners is None:
            self.inhibition = False
            self.num_winners = self.n
        else:
            self.inhibition = True
            self.num_winners = num_winners  
        # There already is a bytetensor in the base class called s for spikes
        # but having a buffer for output sums is still helpful if applying inhibition:
        self.register_buffer("output_sums", torch.zeros(self.shape, dtype=torch.int8))
        self.register_buffer("output_history", torch.zeros((self.timesteps, *self.shape), dtype=torch.int8))
        self.counter = 0 # To keep track of time

    # Forward function needs to sum across the time domain:
    # x_temporal is a tensor of post-synaptic activations.
    # This means that each single spike (if present) along the time 
    # dimension has already been scaled by the weight for that 
    # input -> neuron connection, and then reduced into a vector.
    # Since this class handles multiple neurons, the input tensor
    # has dimensionality num_neurons x time.
    def forward(self, x) -> None:
        # print("COUNTER IN TNN LAYER: ", self.counter)
        # self.v - voltages
        # self.thresh - firing threshold
        # self.s - self output (temporal coded)
        # Where are the weights? These are presynaptic. The x_temporal we 
        # receive here is post synaptic, and already scaled by synaptic weight.
        # Therefore each neuron just needs to integrate over time and threshold.
        self.cumulative_inputs = torch.squeeze(x)
        self.output_history[self.counter,(self.cumulative_inputs >= self.threshold)] = 1
        self.counter += 1
        self.pointwise_inhibition() # Apply inhibition to self.s 
        if self.counter == self.timesteps:
            self.wave_reset()
        super().forward(x)

    def wave_reset(self):
        self.cumulative_inputs.zero_()
        self.output_history.zero_()
        self.output_sums.zero_()
        self.counter = 0
        self.s.zero_()

    def reset_state_variables(self) -> None:
        self.wave_reset()
        super().reset_state_variables()

# Apply pointwise inhibition to computed spike outputs
    def pointwise_inhibition(self) -> None:
        self.output_sums = torch.squeeze(torch.sum(self.output_history, 0))
        if not self.inhibition or self.num_winners >= self.n:
            self.s[(self.output_sums >= 1).unsqueeze(0)] = 1

        # check how many more winners need to be selected
        num_more_winners = self.num_winners - self.s.nonzero().size()[0]
        if(num_more_winners>=0):
            idx = (self.s==0).nonzero()[:,1] # get indices that are not already winners
            flattened_spikes = torch.flatten(self.output_sums[idx])
            max_val = flattened_spikes.max(dim=0)[0] # get the spike time of the winner
            idx = (flattened_spikes==max_val).nonzero() # get indices of possible winners
            winner_idx = self.s.size()[0]
            if(max_val > 0 and idx.flatten().size()[0] <= num_more_winners):
                winner_idx = idx.flatten().squeeze()
                self.s[:,winner_idx] = 1
            elif(max_val > 0):
                winner_idx = np.random.choice(idx.flatten(),replace=False,size=num_more_winners) # randomly choose a winner
                self.s[:,winner_idx] = 1        

            

        

        
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
        maxweight: float, 
        timesteps: int = None,
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
        self.counter = 0
        self.timesteps = timesteps
        #trand.manual_seed(0)        # set seed for determinism
        self.input_spikes = torch.zeros((timesteps, *self.connection.source.shape) , dtype=torch.int)
        self.output_spikes = torch.zeros((timesteps, *self.connection.target.shape), dtype=torch.int)


    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        #print("COUNTER IN UPDATE: ", self.counter)
        #print(self.input_spikes.shape)
        #print(self.output_spikes.shape)

        #print(torch.flatten(self.connection.source.s).shape)
        #print(torch.flatten(self.connection.target.s).shape)
        self.input_spikes[self.counter, :] = torch.flatten(self.connection.source.s)
        self.output_spikes[self.counter, :] = torch.flatten(self.connection.target.s)
        self.counter += 1
        if (self.counter == self.timesteps):
            self.counter = 0
            weights = self.connection.w.view( (self.connection.source.n, self.connection.target.n) )
            old_weights = torch.clone(weights)
            # Copied from lab 1 in spyketorch:
            # Find when input spike occurred (if at all) for each input channel 
            # and receptive field row and column:
            x_times = torch.flatten(int(self.maxweight) - torch.sum(self.input_spikes.int(), 0))
            # ^ should have shape connection.source.shape

            # Do the same for outputs (max weight = number of time steps):
            y_times = torch.flatten(int(self.maxweight) - torch.sum(self.output_spikes.int(), 0))
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
            # neuron and vice-versa. This is sort of an outer product?
            desired_size = (weights.size())
            bcast_y = torch.zeros(desired_size)
            bcast_x = torch.zeros(desired_size)
            # For each elem in y, x needs to be copied across the 2nd dimension of bcast_x 
            for i in range( y_slice.numel() ):
                bcast_x[:,i] = x_slice

            # and for each elem in x, y must be copied across the 1st dimension of bcast_y
            for i in range(x_slice.numel()):
                bcast_y[i,:] = y_slice

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

            
            # Need 2 sets of probabilities for increment and decrement:
            probs_plus = torch.zeros_like(weights).float()
            probs_minus = torch.zeros_like(weights).float()
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
            F_probs_ratio = torch.mul(weights, inv_max_weight)
            #print("PROBS:")
            #print(F_probs_ratio)
            #print("WEIGHTS:")
            #print(weights)

            # Compute F +/- probabilities 
            F_minus_probs = (1-F_probs_ratio) * (1+F_probs_ratio)
            F_minus = torch.bernoulli(F_minus_probs)
            F_plus_probs = F_probs_ratio * (2 - F_probs_ratio)
            F_plus = torch.bernoulli(F_plus_probs)

            # add umin probability to F+/- probability:
            umin_bernoulli = torch.bernoulli(umin_tensor)
            F_plus = torch.max(F_plus, umin_bernoulli)
            F_minus = torch.max(F_minus, umin_bernoulli)

            # Apply updates to weights (ewise add)
            weights.add_(bernoulli_frame_plus * F_plus)
            weights.add_(-1 * bernoulli_frame_minus * F_minus)

            # Clamp outputs to range

            weights.clamp_(0, self.maxweight)

            #print(old_weights)
            #print(weights)
            #print(self.connection.w)
            # pdb.set_trace()
            # self.connection.w = weights
            # Assign updated weights back to layer
            # self.connection.w = weights_patch.int()

        super().update()



def ramp_no_leak(
    datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Encodes data via a temporal (thermometer code) ramp-no-leak specification. 
    One spike per neuron, but on for the rest of the timeline,
    temporally ordered by decreasing intensity. Inputs must be non-negative.

    :param datum: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.flatten()
    timesteps = int(time / dt)

    # Create spike times in order of decreasing intensity.
    datum /= datum.max()
    times = torch.ones(size) * timesteps
    times -= torch.ceil(datum*timesteps).long()

    # Create spike times tensor.
    spikes = torch.zeros(time, size).byte()
    for i in range(timesteps):
        spikes[i, times <= i] = 1

    return spikes.reshape(time, *shape)

'''
def ramp_no_leak_preproc(
    datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Encodes data via a temporal (thermometer code) ramp-no-leak specification. 
    One spike per neuron, but on for the rest of the timeline,
    temporally ordered by decreasing intensity. Inputs must be non-negative.

    :param datum: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    preproc = kwargs['preproc']
    preproc_datum = preproc(datum)
    assert (preproc_datum >= 0).all(), "Inputs must be non-negative"

    shape, size = preproc_datum.shape, preproc_datum.numel()
    preproc_datum = preproc_datum.flatten()
    timesteps = int(time / dt)

    # Create spike times in order of decreasing intensity.
    preproc_datum /= preproc_datum.max()
    times = torch.ones(size) * timesteps
    times -= torch.ceil(preproc_datum*timesteps).long()

    # Create spike times tensor.
    spikes = torch.zeros(time, size).byte()
    for i in range(timesteps):
        spikes[i, times <= i] = 1

    return spikes.reshape(time, *shape)
'''

'''
# Kernel filters adapted from Hari's modifications to Spyketorch
# returns a 2d tensor corresponding to the requested On filter
def getOnKernel(window_size=3):
    #onfilter = np.array([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
    mid = int((window_size-1)/2)
    onfilter = np.ones((window_size,window_size))*-1.0
    onfilter[mid][mid] = (window_size**2)-1
    ontensor = torch.from_numpy(onfilter)
    return ontensor.float()

# returns a 2d tensor corresponding to the requested Off filter
def getOffKernel(window_size=3):
    #offfilter = np.array([[1.0,1.0,1.0],[1.0,-8.0,1.0],[1.0,1.0,1.0]])
    mid = int((window_size-1)/2)
    offfilter = np.ones((window_size,window_size))
    offfilter[mid][mid] = 1-(window_size**2)
    offtensor = torch.from_numpy(offfilter)
    return offtensor.float()


# Adapted from spyketorch
class TNNFilter:
    r"""Applies a filter transform. Each filter contains a sequence of :attr:`FilterKernel` objects.
    The result of each filter kernel will be passed through a given threshold (if not :attr:`None`).

    Args:
        filter_kernels (sequence of FilterKernels): The sequence of filter kernels.
        padding (int, optional): The size of the padding for the convolution of filter kernels. Default: 0
        thresholds (sequence of floats, optional): The threshold for each filter kernel. Default: None
        use_abs (boolean, optional): To compute the absolute value of the outputs or not. Default: False

    .. note::

        The size of the compund filter kernel tensor (stack of individual filter kernels) will be equal to the 
        greatest window size among kernels. All other smaller kernels will be zero-padded with an appropriate 
        amount.
    """
    # filter_kernels must be a list of filter kernels
    # thresholds must be a list of thresholds for each kernel
    def __init__(self, filter_kernels, padding=0, thresholds=None, use_abs=False):
        tensor_list = []
        self.max_window_size = 0
        for kernel in filter_kernels:
            if isinstance(kernel, torch.Tensor):
                tensor_list.append(kernel)
                self.max_window_size = max(self.max_window_size, kernel.size(-1))
            else:
                tensor_list.append(kernel().unsqueeze(0))
                self.max_window_size = max(self.max_window_size, kernel.window_size)
        for i in range(len(tensor_list)):
            if isinstance(kernel, torch.Tensor):
                p = (self.max_window_size - filter_kernels[i].size(-1))//2
            else:
                p = (self.max_window_size - filter_kernels[i].window_size)//2
            tensor_list[i] = fn.pad(tensor_list[i], (p,p,p,p))

        self.kernels = torch.stack(tensor_list)
        self.number_of_kernels = len(filter_kernels)
        self.padding = padding
        if isinstance(thresholds, list):
            self.thresholds = thresholds.clone().detach()
            self.thresholds.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
        else:
            self.thresholds = thresholds
        self.use_abs = use_abs

    # returns a 4d tensor containing the flitered versions of the input image
    # input is a 4d tensor. dim: (minibatch=1, filter_kernels, height, width)
    def __call__(self, input):
        print(self.kernels)
        print(self.padding)
        output = fn.conv2d(input.unsqueeze(0), self.kernels, padding = self.padding).float()
        if not(self.thresholds is None):
            output = torch.where(output < self.thresholds, torch.tensor(0.0, device=output.device), output)
        if self.use_abs:
            torch.abs_(output)
        return output
'''
class RampNoLeakTNNEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable Ramp-No-Leak Encoder which 
        temporally encodes inputs as defined in
        :code:`ramp_no_leak`

        :param time: Length of RampNoLeak spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = ramp_no_leak
'''
class OnOffTNNEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        # language=rst
        """
        Creates a callable Ramp-No-Leak Encoder which 
        temporally encodes inputs as defined in
        :code:`ramp_no_leak`

        :param time: Length of RampNoLeak spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)
        self.preproc = kwargs['preproc']
        if self.preproc is None:
            self.enc = ramp_no_leak
        else:
            self.enc = ramp_no_leak_preproc
'''
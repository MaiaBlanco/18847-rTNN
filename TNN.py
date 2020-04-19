import pdb
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
from bindsnet.network.topology import AbstractConnection, Connection
from bindsnet.network.monitors import AbstractMonitor
from bindsnet.learning.reward import AbstractReward
from bindsnet.network.network import Network
from bindsnet.learning import LearningRule

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
        self.register_buffer("buffer1", 
            torch.zeros((self.timesteps, *self.shape), dtype=torch.int8))
        self.register_buffer("buffer2", 
            torch.zeros((self.timesteps, *self.shape), dtype=torch.int8))
        self.tick=False
        self.counter = 0 # To keep track of time

    def forward(self, x) -> None:
        self.s.zero_()
        if not self.tick:
            self.s[:,self.buffer2[self.counter,:] > 0] = 1
            self.buffer1[self.counter,:] = x.clone()
        else:
            self.s[:,self.buffer1[self.counter,:] > 0] = 1
            self.buffer2[self.counter,:] = x.clone()
        print()
        print("X: ",x)
        print("S: ", self.s)
        self.counter += 1
        if (self.counter == self.timesteps):
            print("BUF1: ", self.buffer1)
            print("BUF2: ", self.buffer2)
        super().forward(x)

    def reset_state_variables(self) -> None:
        self.tick = not self.tick
        self.counter = 0
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
        self.cumulative_inputs += torch.squeeze(x)
        self.output_history[self.counter,(self.cumulative_inputs >= self.threshold)] = 1
        self.counter += 1
        # print(torch.flatten(x))
        # print(self.cumulative_inputs)
        # print(self.output_history)
        if (self.counter == self.timesteps):
            self.pointwise_inhibition() # Apply inhibition to self.s 
            #print(self.s)
        super().forward(x)

    def reset_state_variables(self) -> None:
        self.cumulative_inputs.zero_()
        self.output_history.zero_()
        self.output_sums.zero_()
        self.counter = 0
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

            print(self.output_sums)            
            print(winner_idx)
            print(self.s)
            input()
        

            

        

        
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

# '''
# Modified network class from bindsnet.
# The modification here is that it detects if a source set of neurons is temporal, 
# and passes along the correct spike set at each simulation time step.
# '''
# class TemporalNetwork(Network):
#     def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
#         inputs = {}

#         if layers is None:
#             layers = self.layers

#         # Loop over network connections.
#         for c in self.connections:
#             if c[1] in layers:
#                 # Fetch source and target populations.
#                 source = self.connections[c].source
#                 target = self.connections[c].target

#                 if not c[1] in inputs:
#                     inputs[c[1]] = torch.zeros(
#                         self.batch_size, *target.shape, device=target.s.device
#                     )

#                 # Add to input: source's spikes multiplied by connection weights.
#                 inputs[c[1]] += self.connections[c].compute(source.s)

#         return inputs

#     def run(
#         self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
#     ) -> None:
#         # Parse keyword arguments.
#         clamps = kwargs.get("clamp", {})
#         unclamps = kwargs.get("unclamp", {})
#         masks = kwargs.get("masks", {})
#         injects_v = kwargs.get("injects_v", {})

#         # Compute reward.
#         if self.reward_fn is not None:
#             kwargs["reward"] = self.reward_fn.compute(**kwargs)

#         # Dynamic setting of batch size.
#         if inputs != {}:
#             for key in inputs:
#                 # goal shape is [time, batch, n_0, ...]
#                 if len(inputs[key].size()) == 1:
#                     # current shape is [n_0, ...]
#                     # unsqueeze twice to make [1, 1, n_0, ...]
#                     inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
#                 elif len(inputs[key].size()) == 2:
#                     # current shape is [time, n_0, ...]
#                     # unsqueeze dim 1 so that we have
#                     # [time, 1, n_0, ...]
#                     inputs[key] = inputs[key].unsqueeze(1)

#             for key in inputs:
#                 # batch dimension is 1, grab this and use for batch size
#                 if inputs[key].size(1) != self.batch_size:
#                     self.batch_size = inputs[key].size(1)

#                     for l in self.layers:
#                         self.layers[l].set_batch_size(self.batch_size)

#                     for m in self.monitors:
#                         self.monitors[m].reset_state_variables()

#                 break

#         # Effective number of timesteps.
#         timesteps = int(time / self.dt)

#         # Simulate network activity for `time` timesteps.
#         for t in range(timesteps):
#             # Get input to all layers (synchronous mode).
#             current_inputs = {}
#             if not one_step:
#                 current_inputs.update(self._get_inputs())

#             for l in self.layers:
#                 # Update each layer of nodes.
#                 if l in inputs:
#                     if l in current_inputs:
#                         current_inputs[l] += inputs[l][t]
#                     else:
#                         current_inputs[l] = inputs[l][t]

#                 if one_step:
#                     # Get input to this layer (one-step mode).
#                     current_inputs.update(self._get_inputs(layers=[l]))

#                 self.layers[l].forward(x=current_inputs[l])

#                 # Clamp neurons to spike.
#                 clamp = clamps.get(l, None)
#                 if clamp is not None:
#                     if clamp.ndimension() == 1:
#                         self.layers[l].s[:, clamp] = 1
#                     else:
#                         self.layers[l].s[:, clamp[t]] = 1

#                 # Clamp neurons not to spike.
#                 unclamp = unclamps.get(l, None)
#                 if unclamp is not None:
#                     if unclamp.ndimension() == 1:
#                         self.layers[l].s[unclamp] = 0
#                     else:
#                         self.layers[l].s[unclamp[t]] = 0

#                 # Inject voltage to neurons.
#                 inject_v = injects_v.get(l, None)
#                 if inject_v is not None:
#                     if inject_v.ndimension() == 1:
#                         self.layers[l].v += inject_v
#                     else:
#                         self.layers[l].v += inject_v[t]

#             # Run synapse updates.
#             for c in self.connections:
#                 self.connections[c].update(
#                     mask=masks.get(c, None), learning=self.learning, **kwargs
#                 )

#             # Get input to all layers.
#             current_inputs.update(self._get_inputs())

#             # Record state variables of interest.
#             for m in self.monitors:
#                 self.monitors[m].record()

#         # Re-normalize connections.
#         for c in self.connections:
#             self.connections[c].normalize()


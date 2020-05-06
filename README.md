# 18847-rTNN
This repository is an archive of work done in exploring recurrent temporal neural networks (rTNNs) in 18847-Neuromorphic Computing Architecture.

The top-level directories are:
* SpykeTorch
* bindsnet_reference
* recurrent_only_scripts - contains variants of recurrent-only architecture
* reference_files
* tnn_readout

Additional top level files are:
* TNN.py - implementations of our extensions to BindsNET for TNN simulation
* TNN_utils.py - utilities used in implementations of recurrent-only rTNN architecture
* rc_template-seq.py - reservoir computing template
* rc_template.py
* rc_template_buff_seq.py - reservoir computing template using TNN neurons and buffer nodes
* stateful_tnn.py - implementation of stateful rTNN architecture
* temp_lif_seq_mnist.py - ??
* test_TNN.py
* test_TNN_2layer.py - implementation of 2-layer rTNN architecture
* test_TNN_lr_readout.py
* test_TNN_w_on_off.py
* test_buffer_nodes.py
* test_inhibit_recur_TNN.py
* tnn_reservoir_1024.py
* tnn_reservoir_excit_inhib.py

Running these codes requires an installation of BindsNET. More details on that can be found here: <https://github.com/BindsNET/bindsnet>.

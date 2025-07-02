from .layers import *
import numpy as np


def map_layers_to_cores(layers, arch, max_neurons):
    """
    Maps neurons from a list of layers to the available cores in the architecture,
    respecting the neuron capacity of each core.

    Args:
        layers (list): A list of Layer objects to be mapped.
        arch (sanafe.Architecture): The hardware architecture object.
        max_neurons (int): The maximum number of neurons each core can support.
                           NOTE: This assumes all cores have the same capacity.
    """
    # print("--- Starting Intelligent Neuron Mapping ---")

    cores = arch.cores()
    if not cores:
        raise ValueError("Architecture has no cores to map neurons to.")

    core_idx = 0
    neurons_on_core = 0
    total_neurons_mapped = 0

    for layer in layers:
        # print(f"Mapping layer '{layer.group.get_name()}' with {len(layer)} neurons...")
        for neuron in layer:
            # Check if the current core is full
            if neurons_on_core >= max_neurons:
                # Move to the next core
                core_idx += 1
                neurons_on_core = 0

                # print(f"Core {core_idx-1} is full. Moving to core {core_idx}.")

                # Check if we have run out of cores
                if core_idx >= len(cores):
                    raise ValueError(
                        f"Ran out of cores to map neurons. "
                        f"Mapped {total_neurons_mapped} neurons, but more are left. "
                        f"Total cores available: {len(cores)}."
                    )

            # Map the neuron to the current core
            neuron.map_to_core(cores[core_idx])
            neurons_on_core += 1
            total_neurons_mapped += 1

    # print(f"--- Finished Mapping ---")
    # print(f"Total neurons mapped: {total_neurons_mapped}")
    # print(f"Total cores used: {core_idx + 1}/{len(cores)}")


def create_scnn(snn, arch, input_config, conv_configs, max_neurons):
    """
    Create a spiking convolutional neural network with
    an input layer and multiple convolutional layers.

    Args:
        snn (sanafe.Network): The SANA-FE network object.
        arch (sanafe.Architecture): The hardware architecture object.
        input_config (dict): The configuration for the input layer.
        conv_configs (list): A list of dictionaries, where each dictionary
                             defines a convolutional layer with its 'weights',
                             'biases' (optional), and other parameters.
        max_neurons (int): The maximum number of neurons each core can support.
    """
    Ts, Ci, Hi, Wi = input_config['inputs'].shape
    reshaped_input = input_config['inputs'].reshape(Ts, -1)

    # Separate kwargs for the input layer using prefixes.
    input_kwargs = {k: v for k, v in input_config.items() if k not in ['inputs']}

    # Create Input Layer
    input_layer = Input2D(snn, width=Wi, height=Hi, channels=Ci, **input_kwargs)
    for i, neuron in enumerate(input_layer):
        neuron.set_attributes(
            soma_attributes={"spikes": reshaped_input[:, i].tolist()},
        )

    prev_layer = input_layer
    all_layers = [input_layer]

    for config in conv_configs:
        # Get weights and biases from the config
        weights = config.pop('weights')
        biases = config.pop('biases', None)
        stride_width, stride_height = config.pop('stride', (1, 1))
        pad_width, pad_height = config.pop('padding', (0, 0))

        # Create the convolutional layer
        conv_layer = Conv2D(snn, prev_layer, weights, biases, 
                            stride_width, stride_height,
                            pad_width, pad_height, **config)

        # Append the layer to the list of all layers
        all_layers.append(conv_layer)
        prev_layer = conv_layer

    map_layers_to_cores(all_layers, arch, max_neurons)

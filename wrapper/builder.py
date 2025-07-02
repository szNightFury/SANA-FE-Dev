from .layers import *
import numpy as np


def create_snn(snn, arch, input_data, conv_layers_config, **kwargs):
    """
    Create a spiking neural network with an input layer and multiple convolutional layers.

    Args:
        snn (sanafe.Network): The SANA-FE network object.
        arch (sanafe.Architecture): The hardware architecture object.
        input_data (np.ndarray): The input spike sequence.
        conv_layers_config (list): A list of dictionaries, where each dictionary
                                   defines a convolutional layer with its
                                   'weights', 'biases' (optional), and other
                                   parameters.
        **kwargs: Additional keyword arguments for layer attributes.
    """
    Ts, Ci, Hi, Wi = input_data.shape
    reshaped_input = input_data.reshape(Ts, -1)

    # Separate kwargs for the input layer using prefixes.
    input_kwargs = {k.replace('input_', ''): v for k, v in kwargs.items() if k.startswith('input_')}

    input_layer = Input2D(snn, width=Wi, height=Hi, channels=Ci, **input_kwargs)
    for i, neuron in enumerate(input_layer):
        neuron.set_attributes(
            soma_attributes={"spikes": reshaped_input[:, i].tolist()},
        )

    prev_layer = input_layer
    all_layers = [input_layer]

    for i, conv_config in enumerate(conv_layers_config):
        # Separate kwargs for the conv layer using prefixes.
        conv_kwargs = {}
        prefix = f'conv{i}_'

        for k, v in kwargs.items():
            if not k.startswith(prefix):
                continue
            key = k[len(prefix):]

            if key.startswith('soma_') and key != 'soma_hw_name':
                sub_key = key[len('soma_'):]
                conv_kwargs.setdefault('soma', {})[sub_key] = v
            elif key.startswith('dendrite_') and key != 'dendrite_hw_name':
                sub_key = key[len('dendrite_'):]
                conv_kwargs.setdefault('dendrite', {})[sub_key] = v
            else:
                conv_kwargs[key] = v

        # Get weights and biases from the config
        weights = conv_config['weights']
        biases = conv_config.get('biases') # It's optional

        # Create the convolutional layer
        conv_layer = Conv2D(snn, prev_layer, weights, biases, **conv_kwargs)
        
        all_layers.append(conv_layer)
        prev_layer = conv_layer

    # Map all neurons to the same core for simplicity
    core = arch.tiles[0].cores[0]
    for layer in all_layers:
        for neuron in layer:
            neuron.map_to_core(core)
import os
import yaml
import numpy as np
from pathlib import Path
from .builder import *
from .sanafecpp import load_arch, Network, SpikingChip


# 1. Load the hardware architecture
arch = load_arch(Path(__file__).parent.parent / "arch" / "loihi.yaml")

# 2. Create the SNN Network object
snn = Network()

# 3. Define the input dimension
Ts, Ci, Hi, Wi = 4, 3, 32, 32

# 4. Define the layer configurations in a structured list of dictionaries
input_config = {
    'inputs': np.random.choice([True, False], size=(Ts, Ci, Hi, Wi)),
    'soma_hw_name': 'loihi_inputs',
    'log_spikes': True
}

conv_config = [
    {
        'weights': np.random.randn(3, 3, 3, 16),
        'biases': np.random.randn(16),
        'stride': (1, 1),
        'padding': (1, 1),
        'default_synapse_hw_name': 'loihi_conv_synapse',
        'soma_hw_name': 'loihi_lif',
        'soma': {
            'threshold': 0.,
            'leak_decay': .5,
            'input_decay': .5,
            'reset_mode': "hard",
            'reset': 0.
        },
        'dendrite': {'leak_decay': 0.},
        'log_spikes': True,
        'log_potential': True,
    },
    {
        'weights': np.random.randn(3, 3, 16, 64),
        'biases': np.random.randn(64),
        'stride': (1, 1),
        'padding': (1, 1),
        'default_synapse_hw_name': 'loihi_conv_synapse',
        'soma_hw_name': 'loihi_lif',
        'soma': {
            'threshold': 0.,
            'leak_decay': .5,
            'input_decay': .5,
            'reset_mode': "hard",
            'reset': 0.
        },
        'dendrite': {'leak_decay': 0.},
        'log_spikes': True,
        'log_potential': True,
    }
]


# 5. Build the network using the flexible wrapper function
create_scnn(snn, arch, input_config, conv_config, max_neurons=1024)

# 6. Run the simulation
chip = SpikingChip(arch)
chip.load(snn)
results = chip.sim(Ts + len(conv_config), spike_trace=False, potential_trace=False,
                   message_trace=False, perf_trace=False)

# 7. Print and analyze the results
print("--- Simulation Results ---")
print(yaml.dump(results, default_flow_style=False))
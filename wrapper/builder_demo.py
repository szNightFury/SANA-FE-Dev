import os
import yaml
import numpy as np
from .builder import *
from .sanafecpp import load_arch, Network, SpikingChip


# 1. Load the hardware architecture.
arch = load_arch(os.path.join(os.path.dirname(__file__), "../arch", "builder_demo.yaml"))
# arch = load_arch("sanafe/arch/builder_demo.yaml")

# 2. Create the SNN Network object.
snn = Network()

# 3. Define the custom input.
# A 10x2x4x4 input over 10 timesteps.
custom_input = np.ones((10, 2, 4, 4), dtype=bool)

# 4. Define the convolutional layers configurations.
conv_layers = [
    {
        'weights': np.random.randn(3, 3, 2, 4),  # First conv layer
        'biases': np.random.randn(4)
    },
    {
        'weights': np.random.randn(2, 2, 4, 8),  # Second conv layer
        'biases': np.random.randn(8)
    }
]


# 5. Build the network using the flexible wrapper function.
# All configuration is passed via kwargs with clear prefixes.
create_snn(
    snn, arch, custom_input, conv_layers,
    input_soma_hw_name='loihi_inputs',
    input_log_spikes=True,

    # -- Conv Layer 0 Parameters --
    conv0_synapse_hw_name='loihi_conv_synapse',
    conv0_soma_hw_name='loihi_lif',
    conv0_soma_threshold=1.,
    conv0_soma_leak_decay=.5,
    conv0_soma_input_decay=.5,
    conv0_soma_reset_mode="hard",
    conv0_soma_reset=0.,
    conv0_dendrite_leak_decay=0.,
    conv0_log_spikes=True,
    conv0_log_potential=True,

    # -- Conv Layer 1 Parameters --
    conv1_synapse_hw_name='loihi_conv_synapse',
    conv1_soma_hw_name='loihi_lif',
    conv1_soma_threshold=1.,
    conv1_soma_leak_decay=.5,
    conv1_soma_input_decay=.5,
    conv1_soma_reset_mode="hard",
    conv1_soma_reset=0.,
    conv1_dendrite_leak_decay=0.,
    conv1_log_spikes=True,
    conv1_log_potential=True,
)

# 6. Run the simulation.
chip = SpikingChip(arch)
chip.load(snn)
results = chip.sim(11, spike_trace=False, potential_trace=False,
                   message_trace=False, perf_trace=False)

# 7. Print and analyze the results.
print("--- Simulation Results ---")
print(yaml.dump(results, default_flow_style=False))
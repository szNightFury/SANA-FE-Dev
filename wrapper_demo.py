import sanafe
import numpy as np
from wrapper import create_snn
import yaml

# 1. Load the hardware architecture.
arch = sanafe.load_arch("./arch/wrapper_demo.yaml")

# 2. Create the SNN Network object.
snn = sanafe.Network()

# 3. Define the custom input.
# A 10x2x4x4 input over 10 timesteps.
custom_input = np.ones((10, 2, 4, 4), dtype=bool)

# 4. Define the convolutional layers configurations.
conv_layers = [
    {
        'weights': np.ones((3, 3, 2, 4)),  # First conv layer
        'biases': np.array([0.1, -0.5, 0.3, 0.5])
    },
    {
        'weights': np.ones((2, 2, 4, 8))  # Second conv layer
        # No biases for the second layer
    }
]


# 5. Build the network using the flexible wrapper function.
# All configuration is passed via kwargs with clear prefixes.
create_snn(
    snn, arch, custom_input, conv_layers,
    input_soma_hw_name='loihi_inputs',
    input_log_spikes=True,

    # -- Conv Layer 0 Parameters --
    conv0_soma_hw_name='loihi_lif',
    conv0_default_synapse_hw_name='loihi_conv_synapse',
    conv0_threshold=1.,
    conv0_leak_decay=0.5,
    conv0_reset_mode="hard",
    conv0_reset=0.0,
    conv0_log_spikes=True,
    conv0_log_potential=True,

    # -- Conv Layer 1 Parameters --
    conv1_soma_hw_name='loihi_lif',
    conv1_default_synapse_hw_name='loihi_conv_synapse',
    conv1_threshold=1.2,
    conv1_leak_decay=0.2,
    conv1_reset_mode="soft",
    conv1_log_spikes=True,
    conv1_log_potential=True,
)

# 6. Run the simulation.
chip = sanafe.SpikingChip(arch)
chip.load(snn)
results = chip.sim(11, spike_trace=False, potential_trace=False,
                   message_trace=False, perf_trace=False)

# 7. Print and analyze the results.
print("--- Simulation Results ---")
print(yaml.dump(results, default_flow_style=False))
import sanafe
import wrapper.layers

arch = sanafe.load_arch("arch/loihi.yaml")
snn = sanafe.Network()

# Load the convolutional kernel weights, thresholds and input biases from file.
# If using the Docker container, this file is included in the image.
# Otherwise, this file is also hosted on Google Drive and can be downloaded
# prior to running this script
import numpy as np
try:
    snn_attributes = np.load("dvs_challenge.npz")
except FileNotFoundError as exc:
    print(exc)
    print("""
To run this challenge, you need to download the network kernel weights: dvs_challenge.npz, to the tutorial directory.
These weights are hosted online on a shared Google Drive. To download the file with a in Linux run the command:

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WkbJZFasTe-v8vTYXrUaz_1e-p_xHMEj' -O tutorial/dvs_challenge.npz

Or go directly to the drive at: https://drive.google.com/drive/folders/1GzjXAFouakm3b6GcFIHsw67H8t6l3BtY?usp=drive_link
          """)
    exit()

# Convert the DVS gesture categorization model to SANA-FE's SNN format
thresholds = snn_attributes["thresholds"]
biases = snn_attributes["inputs"]

layer0 = sanafe.layers.Input2D(snn, 32, 32, threshold=thresholds[0])
layer1 = sanafe.layers.Conv2D(snn, layer0, snn_attributes["conv1"],
                              stride_width=2, stride_height=2, threshold=thresholds[1])
layer2 = sanafe.layers.Conv2D(snn, layer1, snn_attributes["conv2"], threshold=thresholds[2])
layer3 = sanafe.layers.Conv2D(snn, layer2, snn_attributes["conv3"], threshold=thresholds[3])
layer4 = sanafe.layers.Conv2D(snn, layer3, snn_attributes["conv4"], threshold=thresholds[4])
layer5 = sanafe.layers.Dense(snn, layer4, 11, snn_attributes["dense1"], threshold=thresholds[5])

# Finally set up the inputs
for n, b in zip(layer0, biases):
    n.set_attributes(model_attributes={"bias": b})


# Map the SNN to Loihi cores. Specify the number of cores each layer is evenly
#  mapped across. Feel free to experiment with changing the line below
layer_mapped_core_counts = [1, 4, 16, 16, 4, 1]


# Map neurons, taking into account the number of cores we want to map across
#  each layer
total_cores_mapped = 0

def map_layer_to_cores(layer, cores, core_count):
    global total_cores_mapped
    total_neurons = len(layer)
    neurons_per_core = total_neurons // core_count
    for idx in range(core_count):
        first_nid = idx * neurons_per_core
        is_last = (idx == (core_count-1))
        if is_last:
            neurons_to_map_to_core = layer[first_nid:]
        else:
            last_nid = (idx+1) * neurons_per_core
            neurons_to_map_to_core = layer[first_nid:last_nid]

        for neuron in neurons_to_map_to_core:
            neuron.map_to_core(cores[total_cores_mapped])
        total_cores_mapped += 1
    return

for n in layer0:
    n.map_to_core(arch.tiles[0].cores[0])

cores = arch.cores()
map_layer_to_cores(layer0, cores, layer_mapped_core_counts[0])
map_layer_to_cores(layer1, cores, layer_mapped_core_counts[1])
map_layer_to_cores(layer2, cores, layer_mapped_core_counts[2])
map_layer_to_cores(layer3, cores, layer_mapped_core_counts[3])
map_layer_to_cores(layer4, cores, layer_mapped_core_counts[4])
map_layer_to_cores(layer5, cores, layer_mapped_core_counts[5])


# Run the network you just generated on Loihi
# Comment out this line if you want to stop the simulations running
chip = sanafe.SpikingChip(arch)
chip.load(snn)
results = chip.sim(4)


# Check the runtime results against expected values to make sure nothing got
#  messed up earlier
# expected_firing_neurons = 365277
# if results["neurons_fired"] != expected_firing_neurons:
#     print(f"Error: The total number of neurons spiking was "
#           f"{results['neurons_fired']}, "
#           f"should be {expected_firing_neurons}")
#     print("Somehow you may have changed the functional behavior of the SNN")
#     raise RuntimeError

# The energy-delay product is our final performance metric. See how low you can
#  get this number!
print(f"Energy: {results["energy"]}")
print(f"Simulation time: {results["sim_time"]}")
print(f"Average Power: {results["energy"]["total"] / results["sim_time"]}")
energy_delay_product = results["energy"]["total"] * results["sim_time"]
print(f"Energy-Delay product: {energy_delay_product}")
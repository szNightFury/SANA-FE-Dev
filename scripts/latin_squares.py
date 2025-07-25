"""
Copyright (c) 2025 - The University of Texas at Austin
This work was produced under contract #2317831 to National Technology and
Engineering Solutions of Sandia, LLC which is under contract
No. DE-NA0003525 with the U.S. Department of Energy.

Run Latin Square solver benchmark (CSP solver)
"""
import matplotlib
matplotlib.use('Agg')

import csv
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import networkx as nx
import time

# SANA-FE libraries
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath((os.path.join(SCRIPT_DIR, os.pardir)))
sys.path.insert(0, os.path.join(PROJECT_DIR))
import sanafe

ARCH_FILENAME = "arch/loihi_with_noise.yaml"
LOIHI_CORES = 128
LOIHI_CORES_PER_TILE = 4
LOIHI_TILES = int(LOIHI_CORES / LOIHI_CORES_PER_TILE)
LOIHI_COMPARTMENTS = 1024
#TIMESTEPS = 10
TIMESTEPS = 1024
#TIMESTEPS = 10240

def calculate_graph_index(N, row, col, digit):
    return ((row*N + col)*N) + digit

"""
def latin_square(N, tiles=LOIHI_TILES, cores_per_tile=LOIHI_CORES_PER_TILE,
                 neurons_per_core=LOIHI_COMPARTMENTS):
    # TODO: support this once I can save SNNs again!
    network = sim.Network(save_mappings=True)
    arch = sim.Architecture()
    print(f"Creating WTA networks for {N} digits")
    #G = nx.DiGraph()
    #G.add_nodes_from(range(0, N**3))

    # For every position in the square, create a WTA layer representing all
    #  possible digit choices
    square = []
    for i in range(0, N):
        row = []
        for j in range(0, N):
            wta = sim.create_layer(network, N,
                                   log_spikes=False,
                                   log_potential=False,
                                   force_update=False,
                                   threshold=64.0,
                                   reset=0.0,
                                   leak=1,
                                   reverse_threshold=-2**7 + 1.0,
                                   reverse_reset_mode="saturate",
                                   soma_hw_name="loihi_stochastic_lif",
                                   synapse_hw_name="loihi_sparse_synapse")
            for neuron in wta.neurons:
                neuron.add_bias(1 * 2**7)
            row.append(wta)
        square.append(row)

    # Connect such that every digit in one position inhibits all other digits in
    #  that position
    connections = 0
    for row in range(0, N):
        for col in range(0, N):
            pos = square[row][col]
            for digit in range(0, N):
                pre_neuron = pos.neurons[digit]
                for d in range(0, N):
                    if d != digit:
                        # Add inhibiting connection for all other digits at this
                        #  position
                        post_neuron = pos.neurons[d]
                        pre_neuron.add_connection(post_neuron, -1)
                        i = calculate_graph_index(N, row, col, digit)
                        j = calculate_graph_index(N, row, col, d)
                        # G.add_edge(i, j, weights=-1)
                        connections += 1

                for r in range(0, N):
                    if r != row:
                        # Add inhibiting connection for this digit at all other
                        #  rows
                        dest = square[r][col]
                        post_neuron = dest.neurons[digit]
                        pre_neuron.add_connection(post_neuron, -1)
                        j = calculate_graph_index(N, r, col, digit)
                        # G.add_edge(i, j, weights=-1)
                        connections += 1

                for c in range(0, N):
                    if c != col:
                        # Add inhibiting connection for this digit at other cols
                        dest = square[row][c]
                        post_neuron = dest.neurons[digit]
                        pre_neuron.add_connection(post_neuron, -1)
                        j = calculate_graph_index(N, row, c, digit)
                        # G.add_edge(i, j, weights=-1)
                        connections += 1

    print(f"Latin square network has {connections} connections")
    network_filename = os.path.join("runs", "dse", f"latin_square_N{N}.net")
    network_path = os.path.join(PROJECT_DIR, network_filename)
    network.save(network_path)
"""


def plot_results(N, network_path):
    if N < 4:
        pos = nx.nx_agraph.graphviz_layout(G)
        nx.draw_networkx(G, pos)
        plt.savefig(os.path.join(PROJECT_DIR, "runs/latin/latin_net.png"))

    # Now execute the network using SANA-FE and extract the spike timings
    arch_path = os.path.join(PROJECT_DIR, ARCH_FILENAME)
    arch = sanafe.load_arch(arch_path)
    net = sanafe.load_net(network_path, arch, use_netlist_format=True)
    chip = sanafe.SpikingChip(arch, record_spikes=True, record_potentials=True)
    chip.load(net)

    chip.sim(TIMESTEPS)

    # Use spiking data to create the grid solution produced by the Loihi run
    with open(os.path.join(PROJECT_DIR, "spikes.csv")) as spikes:
        reader = csv.reader(spikes)
        header = next(reader)

        spike_counts = np.zeros((N, N, N))
        for spike in reader:
            gid_str, nid_str = spike[0].split(".")
            gid, nid = int(gid_str), int(nid_str)
            timestep = int(spike[1])

            digit = nid
            col = (gid-1) % N
            row = (gid-1) // N
            assert(0 <= digit < N)
            assert(0 <= col < N)
            assert(0 <= row < N)
            spike_counts[row][col][digit] += 1

    print(spike_counts)
    chosen_digits = np.argmax(spike_counts, axis=2)

    # Plot a grid and fill in the numbers based on the largest number of
    #  spikes collected after a fixed point
    plt.figure(figsize=(1, 1))
    fig, ax = plt.subplots(1, 1)
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=chosen_digits, colWidths=[0.1] * N*N, cellLoc="center",
            loc="center")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "runs/latin/latin_square.png"))

    df = pd.read_csv("potentials.csv")
    plt.figure()
    df.plot(x="timestep")
    plt.savefig(os.path.join(PROJECT_DIR, "runs/latin/latin_potentials.png"))


def run_experiment(network_filename, timing_model):
    arch_path = os.path.join(PROJECT_DIR, ARCH_FILENAME)
    network_path = os.path.join(PROJECT_DIR, network_filename)

    arch = sanafe.load_arch(arch_path)
    net = sanafe.load_net(network_path, arch, use_netlist_format=True)
    chip = sanafe.SpikingChip(arch, record_spikes=True, record_potentials=True,
                              record_messages=True)
    chip.load(net)
    results = chip.sim(TIMESTEPS, timing_model=timing_model)

    return results


if __name__ == "__main__":
    run_experiments = False
    plot_experiment = True

    if run_experiments:
        if (os.path.isfile(os.path.join(PROJECT_DIR, "runs", "latin",
                           "loihi_latin.csv"))):
            with open(os.path.join(PROJECT_DIR, "runs", "latin", "sim_latin.csv"),
                 "w") as latin_squares_file:
                latin_squares_file.write("N,network,sim_energy,sim_latency,sim_cycle\n")

            with open(os.path.join(PROJECT_DIR, "runs", "latin",
                                    "loihi_latin.csv")) as latin_squares_file:
                reader = csv.DictReader(latin_squares_file)

                for line in reader:
                    # Each line of loihi_latin.csv is another experiment,
                    #  containing the network to run and the results measured
                    #  on Loihi
                    results = run_experiment(line["network"], timing_model="detailed")
                    time = results["sim_time"] / TIMESTEPS
                    energy = results["energy"]["total"] / TIMESTEPS
                    # Run the same experiment again using the cycle accurate
                    #  timing model
                    row = [line["N"], line["network"], energy, time]
                    results = run_experiment(line["network"], timing_model="cycle")
                    cycle_accurate = results["sim_time"] / TIMESTEPS
                    row.append(cycle_accurate)

                    with open(os.path.join(PROJECT_DIR, "runs/latin/sim_latin.csv"),
                              "a") as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow(row)
                    # plot_results(int(line["N"]), line["network"])

    if plot_experiment:
        sim_df = pd.read_csv(os.path.join(PROJECT_DIR,
                                          "runs", "latin", "sim_latin.csv"))
        loihi_df = pd.read_csv(os.path.join(PROJECT_DIR,
                                            "runs", "latin", "loihi_latin.csv"))
        df = pd.merge(sim_df, loihi_df)
        sim_energy = df["sim_energy"].values * 1.0e6
        loihi_energy = df["loihi_energy"].values * 1.0e6
        sim_latency = df["sim_latency"].values * 1.0e6
        loihi_latency = df["loihi_latency"].values * 1.0e6
        booksim_latency = df["sim_cycle"].values * 1.0e6
        print(booksim_latency)
        print(loihi_latency)

        # Plot the simulated vs measured energy
        plt.rcParams.update({"font.size": 6, "lines.markersize": 5})
        plt.figure(figsize=(1.5, 1.5))
        plt.minorticks_on()
        plt.gca().set_box_aspect(1)

        plt.plot(sim_energy, loihi_energy, "x", mew=1.5)
        plt.plot(np.linspace(min(sim_energy), max(sim_energy)),
                 np.linspace(min(sim_energy), max(sim_energy)), "k--")
        plt.xlabel("Simulated Energy ($\mu$J)")
        plt.ylabel("Measured Energy ($\mu$J)")
        plt.xticks(np.arange(0, 1.1, 0.4))
        plt.yticks(np.arange(0, 1.1, 0.4))
        plt.tight_layout(pad=0.3)

        plt.savefig(os.path.join(PROJECT_DIR, "runs", "latin",
                                 "latin_energy.pdf"))
        plt.savefig(os.path.join(PROJECT_DIR, "runs", "latin",
                                 "latin_energy.png"))

        # Plot the simulated vs measured latency
        plt.figure(figsize=(1.5, 1.5))
        plt.minorticks_on()
        plt.gca().set_box_aspect(1)

        plt.plot(booksim_latency, loihi_latency, "s", mew=1.5, markerfacecolor="none")
        plt.plot(sim_latency, loihi_latency, "x", mew=1.5)
        plt.plot(np.linspace(min(sim_latency), max(sim_latency)),
                 np.linspace(min(sim_latency), max(sim_latency)), "k--")
        plt.xlabel("Simulated Latency ($\mu$s)")
        plt.ylabel("Measured Latency ($\mu$s)")
        plt.xticks(np.arange(0, 41, 20))
        plt.yticks(np.arange(0, 41, 20))
        plt.tight_layout(pad=0.3)

        plt.savefig(os.path.join(PROJECT_DIR, "runs", "latin",
                                 "latin_latency.pdf"))
        plt.savefig(os.path.join(PROJECT_DIR, "runs", "latin",
                                 "latin_latency.png"))

        absolute_latency_error = np.abs(loihi_latency - sim_latency) / loihi_latency
        absolute_energy_error = np.abs(loihi_energy - sim_energy) / loihi_energy

        print(f"latency absolute mean error: {np.mean(absolute_latency_error) * 100.0}")
        print(f"energy absolute mean {np.mean(absolute_energy_error) * 100.0}")

        total_latency_error = (np.sum(loihi_latency) - np.sum(sim_latency)) / np.sum(loihi_latency)
        total_energy_error = (np.sum(loihi_energy) - np.sum(sim_energy)) / np.sum(loihi_energy)

        print(f"loihi times: {loihi_latency * TIMESTEPS * 1.0E-6}")
        print(f"total latency error: {total_latency_error * 100.0}%")
        print(f"total energy error: {total_energy_error * 100.0}%")

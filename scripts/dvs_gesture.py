"""
Copyright (c) 2025 - The University of Texas at Austin
This work was produced under contract #2317831 to National Technology and
Engineering Solutions of Sandia, LLC which is under contract
No. DE-NA0003525 with the U.S. Department of Energy.

Run DVS Gesture and extract some performance statistics
"""
#import matplotlib
#matplotlib.use('Agg')

import csv
import yaml
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath((os.path.join(SCRIPT_DIR, os.pardir)))
sys.path.insert(0, PROJECT_DIR)
import sanafe

ARCH_FILENAME = "loihi.yaml"
#NETWORK_FILENAME = "dvs_gesture_32x32.net"
NETWORK_FILENAME = "dvs_gesture_32x32.net.tagged"
LOIHI_TIME_DATA_FILENAME = "loihi_gesture_32x32_time.csv"
LOIHI_ENERGY_DATA_FILENAME = "loihi_gesture_32x32_energy.csv"
SIM_TIME_DATA_FILENAME = "sim_gesture_32x32_time.csv"
SIM_ENERGY_DATA_FILENAME = "sim_gesture_32x32_energy.csv"

#NETWORK_DIR = os.path.join(PROJECT_DIR, "runs", "dvs", "loihi_gesture_32x32_apr03")
#NETWORK_DIR = os.path.join(PROJECT_DIR, "runs", "dvs", "loihi_gesture_32x32_sep30")
NETWORK_DIR = os.path.join(PROJECT_DIR, "runs", "dvs", "loihi_gesture_32x32")
DVS_RUN_DIR = os.path.join(PROJECT_DIR, "runs", "dvs")

ARCH_PATH = os.path.join(PROJECT_DIR, "arch", ARCH_FILENAME)
GENERATED_NETWORK_PATH = os.path.join(DVS_RUN_DIR, NETWORK_FILENAME)
#LOIHI_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, "loihi_gesture_32x32_apr03", LOIHI_TIME_DATA_FILENAME)
LOIHI_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, LOIHI_TIME_DATA_FILENAME)
LOIHI_ENERGY_DATA_PATH = os.path.join(DVS_RUN_DIR, LOIHI_ENERGY_DATA_FILENAME)
SIM_TIME_DATA_PATH = os.path.join(DVS_RUN_DIR, SIM_TIME_DATA_FILENAME)
SIM_ENERGY_DATA_PATH = os.path.join(DVS_RUN_DIR, SIM_ENERGY_DATA_FILENAME)


def parse_stats(stats):
    print("Parsing statistics")
    analysis = {}
    analysis["hops"] = stats.loc[:, "hops"]
    analysis["fired"] = stats.loc[:, "fired"]
    analysis["packets"] = stats.loc[:, "packets"]
    analysis["times"] = stats.loc[:, "sim_time"]
    analysis["total_energy"] = sum(stats.loc[:, "total_energy"])

    print("Finished parsing statistics")
    return analysis


def parse_loihi_spiketrains(total_timesteps):
    # Parse the CSV generated from the DVS gesture runs on Loihi
    #  The format is - first line is the neuron ID
    #  Second line is the timestep
    files = ("inputs.csv", "0Conv2D_15x15x16.csv", "1Conv2D_13x13x32.csv",
             "2Conv2D_11x11x64.csv", "3Conv2D_9x9x11.csv", "5Dense_11.csv")

    neurons = []
    timesteps = []

    for i in range(0, len(files)):
        f = files[i]
        path = os.path.join(DVS_RUN_DIR, "spiketrains", f)

        with open(path, "r") as spiketrain:
            reader = csv.reader(spiketrain)

            neurons += next(reader)
            timesteps += next(reader)

    spiketrain = {}
    for t in range(0, total_timesteps+1):
        spiketrain[t] = []

    for n, t in zip(neurons, timesteps):
        # Why are we -2 out of sync?
        #  Need to subtract 1, because the SNN toolbox starts from timestep 1,
        #   whereas my simulator goes from timestep 0
        t = int(t) - 2
        n = int(n)
        spiketrain[t].append(int(n))

    return spiketrain


if __name__ == "__main__":
    run_experiments = True
    plot_experiments = True
    experiment = "time"
    #experiment = "energy"

    neurons = []
    spiking_times = []
    spiking_update_energy = []
    spiking_spike_gen_energy = []
    spiking_synapse_energy = []
    spiking_network_energy = []
    times = np.array(())
    energies = np.array(())
    hops = np.array(())
    timesteps = 128
    #timesteps = 100000
    frames = 100
    #frames = 1

    #loihi_spiketrains = parse_loihi_spiketrains(timesteps)
    if run_experiments:
        neurons = ""
        groups = ""

        # neuron_groups_filename = os.path.join(NETWORK_DIR, "neuron_groups.net")
        # with open(neuron_groups_filename, "r") as group_file:
        #     group_data = group_file.read()

        # snn_filename = os.path.join(NETWORK_DIR, "dvs_gesture.net")
        # with open(snn_filename, "r") as snn_file:
        #     snn_data = snn_file.read()

        # print("Reading mapping file")
        # mappings_filename = os.path.join(NETWORK_DIR, "mappings.net")
        # with open(mappings_filename, "r") as mappings_file:
        #     mapping_data = mappings_file.read()

        print("Reading input CSV file")

        # Clear the data files
        if experiment == "energy":
            open(SIM_ENERGY_DATA_PATH, "w")
        elif experiment == "time":
            open(SIM_TIME_DATA_PATH, "w")

        # input_filename = os.path.join(NETWORK_DIR, "inputs0.net")
        input_csv_filename = os.path.join(NETWORK_DIR, "inputs.csv")
        # with open(input_filename, "r") as input_file:
        #     input_data = input_file.read()
        with open(input_csv_filename, "r") as input_csv:
            inputs = np.loadtxt(input_csv, delimiter=",", skiprows=1)

        # First create the network file from the inputs and SNN
        # data = (group_data + "\n" + input_data + "\n" + snn_data + "\n" +
        #         mapping_data)
        # TODO: clean up this first part of the script which is redundant now
        #  basically
        #with open(GENERATED_NETWORK_PATH, "w") as network_file:
        #    network_file.write(data)

        # Use a pre-generated network for a realistic use case i.e.
        #  dvs-gesture
        arch = sanafe.load_arch(ARCH_PATH)
        net = sanafe.load_net(GENERATED_NETWORK_PATH, arch,
                              use_netlist_format=True)
        #chip = sanafe.SpikingChip(arch, record_perf=True)
        chip = sanafe.SpikingChip(arch)
        chip.load(net)

        #for frame in range(0, 1):
        ##for frame in range(1, 2):
        ##for frame in range(50, frames):
        for frame in range(0, frames):
            print(f"Running for input: {frame}")
            dvs_inputs = inputs[frame, :]
            mapped_inputs = chip.mapped_neuron_groups["0"]
            for id, mapped_neuron in enumerate(mapped_inputs):
                mapped_neuron.set_model_attributes(
                    model_attributes={"bias": dvs_inputs[id]})
            is_first_frame = (frame == 0)
            chip.sim(timesteps, perf_trace="perf.csv",
                     write_trace_headers=is_first_frame)
            chip.reset()

        # Parse the detailed perf statistics
        print("Reading performance data")
        stats = pd.read_csv(os.path.join(PROJECT_DIR, "perf.csv"))
        analysis = parse_stats(stats)
        times = np.append(times, analysis["times"])
        energies = np.append(energies, analysis["total_energy"] / timesteps)
        hops = np.append(hops, analysis["hops"])

        #with open("hops.csv", "a") as hops_file:
        #    np.savetxt("hops.csv", hops, delimiter=",")
        if experiment == "time":
            with open(SIM_TIME_DATA_PATH, "a") as time_file:
                np.savetxt(SIM_TIME_DATA_PATH, times, delimiter=",")
        else:  # energy
            with open(SIM_ENERGY_DATA_PATH, "a") as energy_file:
                np.savetxt(SIM_ENERGY_DATA_PATH, energies,
                            delimiter=",")
        print("Finished running experiments")

    if plot_experiments:
        """
        plt.figure(figsize=(5.5, 5.5))
        plt.bar(1, spiking_update_energy)
        plt.bar(2, spiking_spike_gen_energy, color="orange")
        plt.bar(3, spiking_synapse_energy, color="red")
        plt.bar(4, spiking_network_energy, color="green")
        plt.xticks([1,2,3,4], ["Update", "Spike Gen", "Synapse", "Network"])
        plt.ylabel("Energy (J)")
        plt.xlabel("Operation Type")
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        plt.savefig("energy_breakdown.png")
        """
        # Plot the latency
        if experiment == "time":
            plt.rcParams.update({'font.size': 6, 'lines.markersize': 4})
            times = np.loadtxt(SIM_TIME_DATA_PATH, delimiter=",")
            print("Reading Loihi data")
            loihi_data = pd.read_csv(LOIHI_TIME_DATA_PATH)
            print("Reading simulated data")
            event_based_data = pd.read_csv(os.path.join(PROJECT_DIR, "runs", "noc", "dvs", "event_based_latencies.csv"))
            cycle_based_data = pd.read_csv(os.path.join(PROJECT_DIR, "runs", "noc", "dvs", "cycle_based_latencies.csv"))

            print("Preprocessing data")
            #loihi_times = np.array(loihi_data.loc[:, "spiking"] / 1.0e6)
            loihi_times = np.array(loihi_data.loc[:, :] / 1.0e6)
            event_based_times = np.array(event_based_data.loc[:, :])
            cycle_based_times = np.array(cycle_based_data.loc[:, :])

            total_loihi_times = np.sum(loihi_times[0:128,:], axis=0)
            print(f"Total Loihi: {total_loihi_times}")
            print(f"Max Total Loihi: {np.max(total_loihi_times)}")
            print(f"Min Total Loihi: {np.min(total_loihi_times)}")

            # There is a weird effect, that the first sample of all inputs > 1 is
            #  a 0 value. Just ignore the entries for both arrays (so we have
            #  timestep-1)
            times = np.delete(times,
                            list(range(timesteps, timesteps*frames, timesteps)))
            #hops = np.delete(hops,
            #                list(range(timesteps, timesteps*frames, timesteps)))
            #loihi_times = np.delete(loihi_times,
            #                list(range(timesteps, timesteps*frames, timesteps)))

            """
            print("Creating plots")
            plt.figure(figsize=(7, 8))
            plt.subplot(311)
            #FRAMES = 100
            FRAMES = 1
            plt.plot(np.arange(1, ((timesteps-1)*FRAMES+1)), times[0:(timesteps-1)*FRAMES], '-')
            plt.plot(np.arange(1, ((timesteps-1)*FRAMES+1)), np.mean(loihi_times[0:(timesteps-1)*FRAMES], axis=1), '--')
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
            plt.legend(("Simulated", "Measured on Loihi"))
            plt.ylabel("Latency (s)")
            plt.xlabel("Timestep")

            # Also plot underneath the different activity on the chip for both my
            #  simulator and Loihi
            #plt.subplot(312)
            #plt.plot(np.arange(1, timesteps+1), analysis["packets"], marker='x')
            #plt.plot(np.arange(1, timesteps+1), analysis["hops"], marker='x')
            #plt.legend(("Packets Sent", "Total Hops"))
            #plt.xlabel("Timestep")

            #plt.subplot(313)
            #plt.plot(np.arange(1, timesteps+1), analysis["fired"], marker='x')
            # Figure out how many neurons fired in the Loihi data
            #fired_count = [len(loihi_spiketrains[i]) for
            #            i in range(0, len(loihi_spiketrains))]
            #plt.plot(np.arange(1, timesteps+1), fired_count[0:timesteps], marker='x')
            #plt.ylabel("Neurons Fired")
            #plt.xlabel("Timestep")
            #plt.legend(("Simulated", "Measured on Loihi"))

            #print("diff = {}".format(
            #    analysis["fired"] - np.array(fired_count[0:timesteps])))
            plt.savefig("runs/dvs/dvs_gesture_sim_time_stats.pdf")
            """

            # Plot the latency
            print("Plotting latency")
            times = np.loadtxt(SIM_TIME_DATA_PATH, delimiter=",")
            loihi_data = pd.read_csv(LOIHI_TIME_DATA_PATH)
            loihi_times = np.array(loihi_data.loc[:, :] / 1.0e6)
            times = np.delete(times,
                    list(range(timesteps-1, timesteps*frames, timesteps)))
            loihi_times = loihi_times[0:timesteps-1,:]

            total_times = np.zeros(frames)
            total_hops = np.zeros(frames)
            loihi_total_times = np.zeros(frames)
            for i in range(0, frames):
                total_times[i] = np.sum(times[i*(timesteps-1)+1:(i+1)*(timesteps-1)])
                total_hops[i] = np.sum(hops[i*(timesteps-1)+1:(i+1)*(timesteps-1)])
                loihi_total_times[i] = np.sum(loihi_times[0:timesteps-2, i])

            plt.figure(figsize=(7.0, 1.6))
            ##plt.plot(np.arange(1, ((timesteps-1)*frames+1)), times[0:(timesteps-1)*frames], marker='x')
            ##plt.plot(np.arange(1, ((timesteps-1)*frames+1)), loihi_times[0:(timesteps-1), frames], marker='x')
            plt.rcParams.update({'font.size': 6})
            #plt.plot(np.arange(1, timesteps-1), loihi_times[0:(timesteps-2), 0] * 1.0e6, "-")
            #plt.plot(np.arange(1, timesteps-1), times[1:(timesteps-1)] * 1.0e6, "--")
            start_frame = 0
            plt.plot(np.arange(1, timesteps-1), loihi_times[0:timesteps-2, start_frame] * 1.0e6, "-")
            plt.plot(np.arange(1, timesteps-1), times[start_frame*(timesteps-1)+1:(start_frame+1)*(timesteps-1)] * 1.0e6, "--")

            #plt.plot(np.arange(1, timesteps-1), event_based_times[1:(timesteps-1)] * 1.0e6, ":k")
            plt.plot(np.arange(1, timesteps-1), cycle_based_times[0:(timesteps-2)] * 1.0e6, ":r")
            plt.legend(("Measured on Loihi", "SANA-FE predictions", "Event-based predictions"),
                       fontsize=6)
            plt.ylabel("Time-step Latency ($\mu$s)")
            plt.xlabel("Time-step")
            plt.yticks(np.arange(0, 61, 10))
            plt.minorticks_on()
            plt.tight_layout(pad=0.3)
            plt.savefig("runs/dvs/dvs_gesture_sim_time.pdf")
            plt.savefig("runs/dvs/dvs_gesture_sim_time.png")

            # Plot the correlation between simulated and measured time-step latency
            plt.figure(figsize=(1.5, 1.5))
            plt.minorticks_on()
            plt.gca().set_box_aspect(1)
            #plt.plot(times[0:frames*(timesteps-1)], loihi_times[0:frames*(timesteps-1)], "x")

            #average_times = total_times / 128
            #loihi_average_times = loihi_total_times / 128

            average_times = total_times / 127
            loihi_average_times = loihi_total_times / 127
            plt.rcParams.update({'font.size': 6, 'lines.markersize': 2})
            #plt.plot(average_times[0:frames] * 1.0e6, loihi_average_times[0:frames] * 1.0e6, "x")
            #plt.plot(np.linspace(min(average_times) * 1.0e6, max(average_times)) * 1.0e6,
            #         np.linspace(min(average_times) * 1.0e6, max(average_times)) * 1.0e6, "k--")

            #cm = plt.colormaps['coolwarm']

            #print(total_hops)
            #exit()
            #plt.scatter(average_times[0:frames]*1.0e6, loihi_average_times[0:frames]*1.0e6, marker="x", s=0.1, cmap=cm, c=np.array(total_hops))
            scatter = plt.plot(average_times[0:frames]*1.0e6, loihi_average_times[0:frames]*1.0e6, "x", alpha=0.5)[0]
            plt.plot(np.linspace(min(average_times)*1.0e6, max(average_times)*1.0e6),
                     np.linspace(min(average_times)*1.0e6, max(average_times)*1.0e6), "k--")
            #plt.colorbar(label="Total Hops", shrink=0.5)
            #plt.xticks((1.0e-5, 1.5e-5, 2.0e-5, 2.5e-5, 3.0e-5))
            #plt.yticks((1.0e-5, 1.5e-5, 2.0e-5, 2.5e-5, 3.0e-5))
            plt.ylabel("Measured Latency ($\mu$s)")
            plt.xlabel("Simulated Latency ($\mu$s)")
            plt.xlim((10, 30))
            plt.ylim((10, 30))
            plt.xticks(np.arange(10, 31, 10))
            plt.yticks(np.arange(10, 31, 10))
            plt.tight_layout(pad=0.3)
            plt.savefig("runs/dvs/dvs_gesture_sim_correlation.pdf")
            plt.savefig("runs/dvs/dvs_gesture_sim_correlation.png")

            # Calculate total error
            print("Calculating errors")
            relative_error = np.abs(loihi_total_times - total_times) / loihi_total_times
            mean_error = np.sum(relative_error) / len(relative_error)
            print("Time Absolute Mean error: {0} ({1} %)".format(mean_error, mean_error * 100))

            total_error =  (np.sum(loihi_total_times) - np.sum(total_times)) / np.sum(loihi_total_times)
            print("Time Total error: {0} ({1} %)".format(total_error, total_error * 100))

            """
            plt.plot(np.arange(1, timesteps+1), analysis["fired"], marker='x')
            # Figure out how many neurons fired in the Loihi data
            fired_count = [len(loihi_spiketrains[i]) for
                        i in range(0, len(loihi_spiketrains))]
            plt.plot(np.arange(1, timesteps+1), fired_count[0:timesteps], marker='x')
            plt.ylabel("Neurons Fired")
            plt.xlabel("Timestep")
            plt.legend(("Simulated", "Measured on Loihi"))

            print("diff = {}".format(
                analysis["fired"] - np.array(fired_count[0:timesteps])))
            plt.savefig("runs/dvs/dvs_gesture_sim_time2.png")
            """
            def on_click(event):
                if event.inaxes != scatter.axes:
                    return
                # Get the data points
                xdata = scatter.get_xdata()
                ydata = scatter.get_ydata()
                # Calculate distances to all points
                distances = np.sqrt((xdata - event.xdata)**2 + (ydata - event.ydata)**2)
                # Find the closest point within a threshold
                threshold = 0.5  # adjust this value to change click sensitivity
                min_dist_idx = np.argmin(distances)
                if distances[min_dist_idx] < threshold:
                    print(f"Frame: {min_dist_idx}")

        # Connect the click event to the figure
        plt.gcf().canvas.mpl_connect('button_press_event', on_click)

        print("Showing plots")
        plt.show()
        print("Time simulations finished")

        if experiment == "energy":
            plt.rcParams.update({'font.size': 6, 'lines.markersize': 2})
            loihi_data = pd.read_csv(LOIHI_ENERGY_DATA_PATH, delimiter=",")
            loihi_energies = np.array(loihi_data).flatten() * 1.0e6
            energies = np.loadtxt(SIM_ENERGY_DATA_PATH) * 1.0e6
            plt.figure(figsize=(1.7, 1.7))
            plt.minorticks_on()
            plt.gca().set_box_aspect(1)

            plt.plot(energies[0:frames], loihi_energies[0:frames], 'x')
            plt.plot(np.linspace(min(loihi_energies), max(loihi_energies)), np.linspace(min(loihi_energies), max(loihi_energies)), "k--")
            plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
            plt.xlim((1, 4))
            plt.ylim((1, 4))
            plt.ylabel("Measured Energy ($\mu$J)")
            plt.xlabel("Simulated Energy ($\mu$J)")
            plt.xticks(np.arange(1, 4.1, 1))
            plt.yticks(np.arange(1, 4.1, 1))
            plt.tight_layout(pad=0.5)
            plt.savefig("runs/dvs/dvs_gesture_sim_energy.png")
            plt.savefig("runs/dvs/dvs_gesture_sim_energy.pdf")

            relative_error = abs(loihi_energies[0:frames] - energies[0:frames]) / loihi_energies[0:frames]
            mean_error = sum(relative_error) / len(relative_error)
            print(relative_error)
            print(f"Energy Absolute Mean error: {mean_error} ({mean_error*100.0} %)")

            total_error = (sum(loihi_energies[0:frames]) - sum(energies[0:frames])) / sum(loihi_energies[0:frames])
            print(f"Energy Total error: {total_error} ({total_error*100.0} %)")
        """
        plt.figure()
        plt.plot(np.arange(1, timesteps+1), analysis["fired"], marker='x')
        # Figure out how many neurons fired in the Loihi data
        fired_count = [len(loihi_spiketrains[i]) for
                    i in range(0, len(loihi_spiketrains))]
        plt.plot(np.arange(1, timesteps+1), fired_count[0:timesteps], marker='x')
        plt.ylabel("Neurons Fired")
        plt.xlabel("Timestep")
        plt.legend(("Simulated", "Measured on Loihi"))
        print("diff = {}".format(
            analysis["fired"] - np.array(fired_count[0:timesteps])))
        plt.savefig("runs/dvs/dvs_gesture_spikes_i16.png")
        """
        exit()
        # These experiments not used for now
        # Plot the potential probes from simulationm this was used to compared
        #  simulated functional behavior against actual
        layers = ("inputs", "0Conv2D_15x15x16", "1Conv2D_13x13x32",
                "2Conv2D_11x11x64", "3Conv2D_9x9x11", "5Dense_11")
        layer_sizes = (1024, 3600, 5408, 7744, 891, 11)
        thresholds = (255, 293, 486, 510, 1729, 473)
        potential_data = pd.read_csv("probe_potential.csv")

        plt.rcParams.update({'font.size': 12, 'lines.markersize': 5})
        plot_neurons = {"inputs": [], "0Conv2D_15x15x16": [50, 67], "1Conv2D_13x13x32": [], "2Conv2D_11x11x64": [], "3Conv2D_9x9x11": [], "5Dense_11":[]}
        for layer_id, layer in enumerate(layers):
            layer_path = "runs/dvs/potentials/{0}".format(layer)
            if not os.path.exists(layer_path):
                os.makedirs(layer_path)

            for neuron_id in plot_neurons[layer]:
                potentials = potential_data.loc[:, "{0}.{1}".format(
                    layer_id, neuron_id)]
                plt.figure(figsize=(5.5, 5.5))
                plt.plot(np.arange(1, timesteps+1), potentials*64, "-x")
                plt.plot(np.arange(1, timesteps+1),
                        np.ones(timesteps) * thresholds[layer_id] * 64)
                #plt.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
                plt.ylabel("Membrane Potential")
                plt.xlabel("Time-step")
                #plt.ylim((-0.2, 1.0))
                plt.tight_layout()
                plt.savefig("{0}/probe_potential_{1}.png".format(layer_path,
                                                            neuron_id))
                plt.close()

        with open("run_summary.yaml", "r") as results_file:
            results = yaml.safe_load(results_file)
        #network_percentage = (results["network_time"] /
        #                                    results["time"]) * 100.0
        #print("Percentage of time used for only network: {0}".format(
        #      network_percentage))

        #plt.show()

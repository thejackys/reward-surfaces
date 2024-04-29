import pandas
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import os
def plot_traj(csv_file, key="episode_rewards", log_plot=False):
    df = pandas.read_csv(csv_file)
    idxs = np.argsort(df['dim0'].values)  # Convert to NumPy array before sorting
    fig, ax = plt.subplots()
    if log_plot:
        ax.set_yscale('log')
    ax.plot(df['dim0'].values[idxs], df[key].values[idxs])  # Convert to NumPy arrays before indexing
    os.makedirs('vis', exist_ok=True)
    out_fname = "vis/" + "".join([c for c in csv_file if re.match(r'\w+', c)]) + "_" + key + ".png"
    plt.savefig(out_fname)
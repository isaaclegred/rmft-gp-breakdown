#!/usr/bin/env python
import numpy as np
import pandas as pd
import seaborn as sns
import os
import h5py
import matplotlib.pyplot as plt
import bilby

import lwp
import lwp.utils.io as io
import argparse

from lwp import executables
parser = argparse.ArgumentParser()
parser.add_argument("index", type=int, default=0)
runs_to_do = np.arange(11)
if __name__ == "__main__":
    args = parser.parse_args()
    index = runs_to_do[args.index]
    eos_runs_dir = "rmf-1109-000338-bps_2p0_1p3_0p7"
    injection_labels = {i  : f"injection_{i}" for i in range(15)}
    injection_label = injection_labels[index]
    astro_data_and_metadata = executables.get_files.get_astro_samples(
        f"{eos_runs_dir}/{eos_runs_dir}/outdir_{injection_label}/final_result/{injection_label}_data0_0_analysis_H1L1_merge_result.hdf5",
        f"{eos_runs_dir}/{injection_label}_result.csv",
        download_url=None, max_num_pe_samples=6000,
        load_samples_kwargs = {"load_function": io.load_bilby})

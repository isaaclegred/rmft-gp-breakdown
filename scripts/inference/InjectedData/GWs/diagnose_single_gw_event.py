import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import universality
from universality import stats
import bilby
import seaborn as sns
import copy

import os
import h5py




from temperance.plotting import corner
import temperance as tmpy
from temperance.core import result
from temperance.core.result import EoSPosterior
from temperance.sampling import eos_prior



if __name__ ==  "__main__":
    eos_set_label= "1109_maxpc2-1e11_mrgagn_01d000_00d010"
    injection_directory = "rmf-1109-000338-bps_1p5_1p4_0p85"
    gw_event_index = 1
    gw_likelihood_tag = f"injection_{gw_event_index}_result"
    other_weight_columns = [
        result.WeightColumn(column_name, is_log=True) for column_name in
        [
            *[f"logweight_xray_{i}" for i in range(3)],
            *[f"logweight_gw_{j}" for j in range(18)],
            *[f"logweight_radio_{k}" for k in range(2)]
        ] if column_name != f"logweight_gw_{gw_event_index}"
    ]

    "./rmf-1109-000338-bps_2p0_1p4_p85/1109_maxpc2-1e11_mrgagn_01d000_00d010/injection_6/runinfo"
    gw_samples = pd.read_csv(
        f"./{injection_directory}/{eos_set_label}/injection_{gw_event_index}/result/injection_{gw_event_index}_post.csv")
    eos_post = EoSPosterior.from_csv(
        f"../{injection_directory}_{eos_set_label}_post.csv")
    gw_samples=gw_samples.sample(
        10000,
        weights=np.exp(
            np.array(gw_samples[f"logweight"])))
    # nicer_samples = pd.merge(
    #     nicer_samples,
    #     eos_post.samples[["eos", []],
    #     on="eos")
    plottable_samples={}
    # plottable_samples["astro-informed"] = corner.PlottableSamples(
    #         label= eos_set_label + "+ {nicer_likelihood_tag}" + "+ other injections",
    #         samples=nicer_samples,
    #         weight_columns_to_use=[*other_weight_columns],
    #         color="navy",
    #     )
    plottable_samples["gw-only"] = corner.PlottableSamples(
            label= eos_set_label + f"-{gw_likelihood_tag}" ,
            samples=gw_samples,
            weight_columns_to_use=[],
            color="salmon",
        )
    
    plottable_samples["lvk-analysis"] = corner.PlottableSamples(
        label="injected-m-r",
        samples=pd.read_csv(f"./{injection_directory}/injection_{gw_event_index}_result.csv"),
        weight_columns_to_use=[],
        color="black"
    )

    # See ./define_injections_metadata.py
    # you can change which columns and how they are plotted here,
    # see temperance/temperance/plotting/corner.py for docs
    # In this case we are plotting the transition columns.
    plottable_columns = [
        corner.PlottableColumn("m1", r"$M\ [M_{\odot}]$",
                               plot_range=(1.0, 2.1), bandwidth=.003),
        corner.PlottableColumn("Lambda1", r"$\Lambda_1$",
                               plot_range=(0, 10000), bandwidth=100),
        corner.PlottableColumn("m2", r"$M_2\ [M_{\odot}]$",
                               plot_range=(1.0, 2.1), bandwidth=.003),
        corner.PlottableColumn("Lambda2", r"$\Lambda_2$",
                               plot_range=(0, 10000), bandwidth=100)]
    



    # Actually make the corner plot
    # note the priors for each plottable_samples is specified explicitly as a dictionary
    # "truths" or injected_values are kept in a dictionary in the metadata
    corner.corner_samples(
        plottable_samples.values(),
        use_universality=True,
        columns_to_plot=plottable_columns,
        rotate_xticklabels=90,
        figwidth=10,
        levels=[ 0.9],
        figheight=10)
    plt.savefig(f"{injection_directory}/{eos_set_label}-{gw_likelihood_tag}.pdf", bbox_inches="tight")


import pandas as pd
import numpy as np

import temperance.core.result as result
from temperance.core.result import EoSPosterior

def merge_likelihoods(eos_set, nicer_data, gw_events, radio_data,  nicer_likelihood_labels, outpath,
                      nuclear_data=None):
    eos_post = EoSPosterior.from_csv(f"{eos_set}.csv", label=eos_set)
    # x-ray
    for i, nicer_tag in enumerate(nicer_data.keys()):
        nicer_obs_likelihoods = pd.read_csv(nicer_data[nicer_tag])
        likelihood_label = nicer_likelihood_labels[nicer_tag]
        nicer_obs_likelihoods[f"logweight_{likelihood_label}"] = nicer_obs_likelihoods[f"logweight_samples_{i}"]
        if f"logweight_{likelihood_label}" in eos_post.samples.columns:
            raise ValueError("Trying to add a weight column"
                             "that already exists")
        eos_post.add_weight_column(
            result.WeightColumn(f"logweight_{likelihood_label}", is_log=True),
            nicer_obs_likelihoods[["eos", f"logweight_{likelihood_label}"]])
    for i, gw_tag in enumerate(gw_events.keys()):
        gw_obs_likelihoods = pd.read_csv(gw_events[gw_tag])
        likelihood_label = f"gw_{i}"
        gw_obs_likelihoods[f"logweight_{likelihood_label}"] = gw_obs_likelihoods["logmargweight"]
        if f"logweight_{gw_tag}" in eos_post.samples.columns:
            raise ValueError("Trying to add a weight column"
                             "that already exists")
        eos_post.add_weight_column(
            result.WeightColumn(f"logweight_{likelihood_label}", is_log=True),
            gw_obs_likelihoods[["eos", f"logweight_{likelihood_label}"]])
    # RADIO observations are different
    radio_df = pd.read_csv(radio_data)
    radio_likelihood_columns = [column for column in radio_df.columns if "logweight" in column]
    likelihood_labels = []
    for i, likelihood_column in enumerate(radio_likelihood_columns):
        likelihood_label = f"radio_{i}"
        radio_df[f"logweight_{likelihood_label}"] = radio_df[likelihood_column]
        eos_post.add_weight_column(
            result.WeightColumn(f"logweight_{likelihood_label}", is_log=True),
            radio_df[["eos", f"logweight_{likelihood_label}"]])
    # NUCLEAR data same as radio
    if nuclear_data is not None:
        nuclear_df = pd.read_csv(nuclear_data)
        nuclear_likelihood_columns =  [column for column in nuclear_df.columns if "logweight" in column]
        for i, likelihood_column in enumerate(nuclear_likelihood_columns):
            likelihood_label = f"nuclear_{i}"
            nuclear_df[f"logweight_{likelihood_label}"] = nuclear_df[likelihood_column]
            eos_post.add_weight_column(
                result.WeightColumn(f"logweight_{likelihood_label}", is_log=False),
                nuclear_df[["eos", f"logweight_{likelihood_label}"]])
    eos_post.samples = eos_post.samples.map(lambda x: x if not np.isnan(x) else -np.inf)
    eos_post.samples.to_csv(f"{outpath}_{eos_set}_post.csv", index=False)
if __name__ == "__main__":
    nicer_dir = "./XRAY"
    gw_dir = "./GW"
    radio_dir = "./RADIO"
    nuclear_dir = "./NUCLEAR"
    injection_eos_dir = "rmf-1109-000338-bps_2p0_1p3_0p7"
    rmf_eos_set = "1109"
    
    for eos_set in [
            f"{rmf_eos_set}_maxpc2-1e11_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-1e12_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-3e12_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-1e13_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-3e13_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-1e14_mrgagn_01d000_00d010",
            f"rmf_{rmf_eos_set}"
    ]:
        nicer_likelihoods = {
            nicer_tag : f"{nicer_dir}/{injection_eos_dir}/{eos_set}/{nicer_tag}_eos.csv"
            for nicer_tag in [f"{eos_set}_samples_{i}" for i in range(4)] }
        nicer_likelihood_labels = {
            nicer_tag : f"xray_{i}"
            for i, nicer_tag in enumerate([f"{eos_set}_samples_{i}" for i in range(4)]) }

        gw_likelihoods = {
            gw_tag : f"{gw_dir}/{injection_eos_dir}/{eos_set}/{gw_tag}/result/{gw_tag}_eos.csv"
            for gw_tag in [f"injection_{i}" for i in [0, 1,2,3,5]]}
        # For better or worse all the radio likelihoods are in the same file
        radio_likelihoods = f"{radio_dir}/{injection_eos_dir}/{eos_set}.csv"
        nuclear_likelihoods =None #f"{nuclear_dir}/{injection_eos_dir}/{eos_set}/{eos_set}_test.csv"

        merge_likelihoods(eos_set,
                          nicer_data=nicer_likelihoods,
                          gw_events=gw_likelihoods,
                          radio_data=radio_likelihoods,
                          nicer_likelihood_labels=nicer_likelihood_labels,
                          outpath=injection_eos_dir,
                          nuclear_data=nuclear_likelihoods)
        
        
        

#!/usr/bin/env python
import numpy as np
import pandas as pd
import bilby
import branched_interpolator
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import argparse

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int, default=0)
def sample_valid_bns_injection_with_eos(prior_dict, eos, bh_values={"Lambda": lambda m : 0.0}, num_injection=50):
    """
    Sample Injections from a BNS population which can be used in a mock-pe campaign
    """
    branches  = branched_interpolator.get_branches(eos, properties=[key for key in bh_values.keys()])
    interpolators = branched_interpolator.get_macro_interpolators(branches, properties=[key for key in bh_values.keys()])
    properties = prior_dict.sample(num_injection)
    loc = properties["mass_1_source"] > properties["mass_2_source"]
    properties = {key :properties[key][loc] for key in properties.keys()}
    #properties = bilby.gw.conversion.generate_mass_parameters(properties)
    m1 = properties["mass_1_source"]
    m2 = properties["mass_2_source"]
    lambda1 = branched_interpolator.choose_macro_per_m(m1, eos, black_hole_values=bh_values, branches=branches, interpolators=interpolators)
    lambda2 = branched_interpolator.choose_macro_per_m(m2, eos, black_hole_values=bh_values, branches=branches, interpolators=interpolators)
    properties["mass_1_source"] = lambda1["m"]
    properties["mass_2_source"] = lambda2["m"]
    properties["lambda_1"] = lambda1["Lambda"]
    properties["lambda_2"] = lambda2["Lambda"]
    return properties
def get_design_sensitivity_optimal_snr(
        samples, ifo_labels=["H1","L1"],
        waveform="IMRPhenomPv2_NRTidalv2", waveform_approximant="IMRPhenomPv2_NRTidalv2",
        maximum_frequency=4096*7/16, reference_frequency=20, sampling_frequency=4096,
        duration=128, wf_generator=None, asd_file="aplus.txt"):
    ifos = [bilby.gw.detector.networks.get_empty_interferometer(ifo_label) for ifo_label in ifo_labels]
    for ifo in ifos:
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=asd_file)
    ifos = bilby.gw.detector.networks.InterferometerList(ifos)
    if wf_generator is None:
        wf_params= {}
        wf_params["waveform_approximant"] = waveform_approximant
        wf_params["maximum_frequency"] = maximum_frequency
        wf_params["minimum_frequency"] = 23
        wf_generator = bilby.gw.waveform_generator.LALCBCWaveformGenerator(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=2.0 - duration,
            frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
            waveform_arguments=wf_params)
    
    optimal_snr_squared = np.zeros_like(samples["mass_1_source"])
    for i in range(len(optimal_snr_squared)):
        sample = {key : samples[key][i] for key in samples.keys()}
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=sample["geocent_time"] - duration + 2)    
        wf_polarizations=ifos.inject_signal(
            waveform_generator=wf_generator, parameters=sample, raise_error=False)
        optimal_snr_squared[i] = np.sum([
            ifos[j].meta_data["optimal_SNR"]**2 for j in range(len(ifos))], axis=0)
    return optimal_snr_squared
        

def prune_samples(samples, cuts={"SNR":lambda sample : sample["optimal_snr_squared"] > 10.0**2,"mass_ratio":
                                 lambda sample : sample["mass_1_source"] > sample["mass_2_source"]}):
    cuts_passed = np.ones_like(samples["mass_1_source"])
    for cut_key in cuts.keys():
        samples[f"{cut_key}_passed"] = cuts[cut_key](samples)
        cuts_passed *= samples[f"{cut_key}_passed"]
    samples_df = pd.DataFrame(samples)
    samples_df["all_cuts_passed"] = [bool(elt) for elt in cuts_passed]
    return samples_df

    


if __name__ == "__main__":
    args = parser.parse_args()
    index = args.index
    EOS_DIR  = "../EoSs/"
    eos_file = "macro-rmf-1109-000338-bps_2p0_1p3_0p7.csv"
    EoSData_H = pd.read_csv(f'{EOS_DIR}/{eos_file}')
    approximant = "IMRPhenomPv2_NRTidalv2"
    eos = EoSData_H
    prior_dict = bilby.core.prior.PriorDict(filename="sampling.prior")
    samples=  sample_valid_bns_injection_with_eos(
        prior_dict, eos, bh_values=  {"Lambda": lambda m :np.zeros_like(m)})

    plt.scatter(samples["mass_1_source"], samples["lambda_1"], label="m1" )
    plt.scatter(samples["mass_2_source"], samples["lambda_2"], label="m2")
    print(samples["mass_1_source"] - samples["mass_2_source"])
    samples["optimal_snr_squared"] = get_design_sensitivity_optimal_snr(samples,waveform_approximant=approximant)
    samples = prune_samples(samples)
    print("snr vs. d_L and M_c", zip(np.sqrt(samples["optimal_snr_squared"]), "\n",
                                     samples["luminosity_distance"], bilby.gw.conversion.component_masses_to_chirp_mass(samples["mass_1_source"], samples["mass_2_source"])))
    plt.plot(eos["M"], eos["Lambda"], label="eos")
    plt.yscale("log")
    plt.xlabel("m")
    plt.ylabel("lambda")
    plt.legend()
    tag = f"low_mass_nearby-{eos_file.split('.csv')[0]}"
    with open(f"samples_{tag}_{index}.json", 'w+') as file_pointer:
        json.dump({"samples": samples.to_dict("list"), "eos":eos.to_dict("list"),
                   "eos_path": f"{EOS_DIR}/{eos_file}",
                   "prior_dict": prior_dict.__repr__(), 
                   "bilby_version": bilby.__version__,
                   "generated_with":__file__,
                   "waveform_approximant":approximant}, file_pointer, cls=NumpyArrayEncoder)
    plt.savefig(f"sampling_{tag}_{index}.pdf", bbox_inches="tight")

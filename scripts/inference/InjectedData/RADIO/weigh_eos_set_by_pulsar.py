import temperance as tmpy
import temperance.core.result as result
from  temperance.core.result import EoSPosterior
import  temperance.weighing.weigh_by_pulsar  as weigh_by_pulsar

import numpy as np
import pandas as pd

def pulsar_sample_likelihood(mmax_eos, mass_samples):
    return np.array([np.sum(mass_samples < mmax)/len(mass_samples) for mmax in mmax_eos])


def weigh_by_pulsar_data(pulsar_mass_sample_sets, eos_posterior, **kwargs):
    for sample_set in pulsar_mass_sample_sets.keys():
        weigh_by_pulsar.weigh_EoSs_by_mass_measurement(
            eos_posterior, likelihood=lambda mmax : pulsar_sample_likelihood(
                mmax, pulsar_mass_sample_sets[sample_set]["M"]),
            weight_tag = sample_set, **kwargs
        )
if __name__ == "__main__":
    injection_dir = "rmf-1109-000338-bps_2p0_1p3_0p7"
    eos_tag = injection_dir
    rmf_eos_set = "1109" 
    pulsar_mass_sets = {
        f"radio_injection_{i}": pd.read_csv(f"{injection_dir}/{eos_tag}_fake_psr_samples_{i}.csv")
        for i in range(2)}
    for eos_set in [
            f"{rmf_eos_set}_maxpc2-1e11_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-1e12_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-3e12_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-1e13_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-3e13_mrgagn_01d000_00d010",
            f"{rmf_eos_set}_maxpc2-1e14_mrgagn_01d000_00d010",
            f"rmf_{rmf_eos_set}"
    ]:
        eos_posterior = EoSPosterior.from_csv(f"../{eos_set}.csv")
        weigh_by_pulsar_data(pulsar_mass_sets, eos_posterior)
        eos_posterior.samples.to_csv(f"{injection_dir}/{eos_set}.csv", index=False)

import temperance.core.result as result
from temperance.weighing import weigh_by_density_estimate
from temperance.sampling import eos_prior
from temperance.core.result import EoSPosterior

from tqdm import tqdm
import pandas as pd
import numpy as np
import os

class FlatMassPrior:
    def __init__(self, m_min, m_max, seed=None):
        self.m_min = m_min
        self.m_max = m_max
        self.rng = np.random.default_rng(seed)

    def sample(self, size):
        return self.rng.uniform(self.m_min, self.m_max, size)

def get_mass_prior(samples_sets):
    dictionary_of_mass_priors = {}
    for samples_key in samples_sets.keys():
        if np.mean(samples_sets[samples_key]["M"]) < 1.8:
            # Set range based on likelihood to improve sampling efficiency, no Occam factor
            dictionary_of_mass_priors[samples_key] = {
                "m_min" : np.min(samples_sets[samples_key]["M"]) * 0.9,
                "m_max" : np.max(samples_sets[samples_key]["M"]) * 1.1}
    else:
        dictionary_of_mass_priors[samples_key] = {
                "m_min" : 1.0}
    return dictionary_of_mass_priors
        
def marginalize_over_mr_samples(
    mr_samples, num_marginalization_samples=250, nicer_tag="j0740"
):
    marginalizable_samples = mr_samples[["eos", "Mmax", f"logweight_{nicer_tag}"]]
    marginalizable_samples[f"weight_{nicer_tag}"] = np.exp(
        marginalizable_samples[f"logweight_{nicer_tag}"]
    )
    marginalized_weights = marginalizable_samples.groupby("eos").mean()
    marginalized_weights.reset_index(inplace=True)
    marginalized_weights[f"logweight_{nicer_tag}"] = np.log(marginalized_weights[f"weight_{nicer_tag}"])
    return marginalized_weights

def weigh_eoss_by_nicer_samples(
        eos_posterior, eos_prior_set,
        nicer_data, mass_prior_kwargs_set, nicer_tag, outdir):
    for i, eos in enumerate(tqdm(eos_posterior.samples[eos_posterior.eos_column])):
        macro_data = pd.read_csv(eos_prior_set.get_macro_path(int(eos)))
        Mmax_loc = eos_posterior.samples.columns.get_loc("Mmax")
        eos_posterior.samples.iloc[i, Mmax_loc] = np.max(macro_data["M"])
        mass_prior_kwargs = mass_prior_kwargs_set[nicer_tag]
        if "m_max" not in mass_prior_kwargs.keys() and eos_posterior.samples.iloc[i, Mmax_loc] < 1.0  :
            # print(
            #     "warning Mmax < Mmin",
            #     "Mmax is",
            #     eos_posterior.samples.iloc[i, Mmax_loc],
            # )
            # EoS produces no samples in range of likelihood anyway
            eos_posterior.samples.iloc[i, Mmax_loc] = 1.0
            continue
    print(mass_prior_kwargs)
    mr_samples = weigh_by_density_estimate.generate_mr_samples(
        eos_posterior, eos_prior_set, FlatMassPrior, num_samples_per_eos=100,
        mass_prior_kwargs=mass_prior_kwargs 
    )

    sample_weights = np.concatenate(
        [
            weigh_by_density_estimate.weigh_mr_samples(
                mr_samples_set[["M", "R"]], nicer_data, bandwidth_factor=1/100
            )
            for mr_samples_set in [*np.array_split(mr_samples, 10)]
        ]
    )

    mr_samples[f"logweight_{nicer_tag}"] = sample_weights
    mr_samples.to_csv(
        f"{outdir}/{eos_posterior.label}_{nicer_tag}_post.csv", index=False
    )
    eos_post = marginalize_over_mr_samples(mr_samples, nicer_tag=nicer_tag)
    #eos_post["eos"] = eos_posterior.samples["eos"]
    eos_post.to_csv(f"{outdir}/{eos_posterior.label}_{nicer_tag}_eos.csv", index=False)



if __name__ == "__main__":
    main_dir = "rmf-1109-000338-bps_2p0_1p3_0p7"
    eos_lowercase = main_dir
    eos_rmf_set = "1109"
    eos_base_directory = f"/home/isaac.legred/RMFGP/gp-rmf-inference/conditioned-priors/many-draws/{eos_rmf_set}/"
    
    nicer_mr_sets = {
        f"samples_{i}": pd.read_csv(f"./{main_dir}/{eos_lowercase}_fake_nicer_samples_{i}.csv") for i in range(4)}
    mass_prior_kwargs_set = get_mass_prior(nicer_mr_sets)
    for eos_set in [
            f"{eos_rmf_set}_maxpc2-1e11_mrgagn_01d000_00d010",
             f"{eos_rmf_set}_maxpc2-1e12_mrgagn_01d000_00d010",
             f"{eos_rmf_set}_maxpc2-3e12_mrgagn_01d000_00d010",
             f"{eos_rmf_set}_maxpc2-1e13_mrgagn_01d000_00d010",
             f"{eos_rmf_set}_maxpc2-3e13_mrgagn_01d000_00d010",
             f"{eos_rmf_set}_maxpc2-1e14_mrgagn_01d000_00d010",
            f"rmf_{eos_rmf_set}"
    ]:
        outdir = f"./{main_dir}/{eos_set}"
        if not os.path.exists(outdir): 
            os.mkdir(outdir)
        eos_posterior = EoSPosterior.from_csv(f"../{eos_set}.csv", label=eos_set)
        eos_dir = f"{eos_base_directory}/{eos_set}"
        eos_per_dir = 1000
        eos_tag = eos_set

        eos_prior_set = eos_prior.EoSPriorSet(
            eos_dir=eos_dir,
            eos_per_dir=eos_per_dir,
            macro_dir=eos_dir,
            macro_path_template="macro-draw-%(draw)06d.csv",
            eos_column="eos",
        )

        for nicer_tag in nicer_mr_sets:
            nicer_data = nicer_mr_sets[nicer_tag]
            weigh_eoss_by_nicer_samples(
                eos_posterior, eos_prior_set,
                nicer_data, mass_prior_kwargs_set, nicer_tag, outdir)


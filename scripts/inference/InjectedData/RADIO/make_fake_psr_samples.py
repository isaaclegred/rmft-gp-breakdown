import pandas as pd
import numpy as np
import sys
sys.path.append("../")

import branched_interpolator

def make_fake_psr_samples(
        mass, mass_uncertainty, num_samples, seed=None
):
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        mean=np.array([mass]),
        cov=np.diag([mass_uncertainty**2]),
        size=num_samples,
    )
    data = pd.DataFrame(samples, columns=["M"])
    data["Prior"] = 1.0
    return data


if __name__ == "__main__":
    # J0437 potential measurement
    eos_tag = "rmf-1109-000338-bps_2p0_1p3_0p7"
    outdir = eos_tag
    eos = pd.read_csv(f"../EoSs/macro-{eos_tag}.csv")
    M_max = max(eos["M"])
    # Region of mass space that an observed pulsar would be considered
    # "interesting" enough to include in an EoS analysis
    delta_M =  M_max * 0.04
    M_injected = -np.random.rand(2) * (delta_M) + M_max 
    print(M_injected)
    for index, mass in enumerate(M_injected):
        sigma_m = .1 * np.random.rand()
        samples = make_fake_psr_samples(mass, sigma_m, 5000)
        samples.to_csv(f"{outdir}/{eos_tag}_fake_psr_samples_{index}.csv",
                       index=False)

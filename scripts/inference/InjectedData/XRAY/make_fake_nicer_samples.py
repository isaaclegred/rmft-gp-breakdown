import pandas as pd
import numpy as np
import sys
sys.path.append("../")

import branched_interpolator

def make_fake_nicer_samples(
    mass, mass_uncertainty, radius, radius_uncertainty, num_samples, seed=None
):
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(
        mean=np.array([mass, radius]),
        cov=np.diag([mass_uncertainty**2, radius_uncertainty**2]),
        size=num_samples,
    )
    data = pd.DataFrame(samples, columns=["M", "R"])
    data["Prior"] = 1.0
    return data


if __name__ == "__main__":
    # J0437 potential measurement
    eos_tag = "rmf-1109-000338-bps_2p0_1p3_0p7"
    outdir = eos_tag
    eos = pd.read_csv(f"../EoSs/macro-{eos_tag}.csv")
    M_injected = np.array([1.2, 1.37, 1.45, 1.96])
    R_injected  = branched_interpolator.choose_macro_per_m(
        M_injected, eos=eos,
        black_hole_values={"R": lambda m : 2 * 1.477 * m}, only_lambda=False)["R"]
    for index, mass in enumerate(M_injected):
        sigma_m = .3 * np.random.rand()
        sigma_R = 1.0 * np.random.rand()
        samples = make_fake_nicer_samples(mass, sigma_m, R_injected[index],
                                          sigma_R, 10000)
        samples.to_csv(f"{outdir}/{eos_tag}_fake_nicer_samples_{index}.csv",
                       index=False)

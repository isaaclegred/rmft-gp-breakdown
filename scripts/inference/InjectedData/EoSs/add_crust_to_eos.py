import pandas as pd
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import universality

from universality import eos as u_eos

from universality.gaussianprocess.utils import integrate_phi

def stitch_crust_to_eos(crust_model, eos):
    """
    Add a crust to the EoS which has the same adiabatic exponent
    as the crust
    """
    crust_model_Gamma = np.gradient(np.log(crust_model["pressurec2"]),
                                    np.log(crust_model["baryon_density"]))

    max_crust_density = min(eos["baryon_density"])

    max_crust_pressure = min(eos["pressurec2"])
    max_crust_internal_energy = (min(eos["energy_densityc2"]) - max_crust_density)/max_crust_density
    # Integrate back from core
    crust_densities = np.geomspace(max_crust_density, 1e1, 100)
    crust_Gamma = interpolate.griddata(crust_model["baryon_density"],
                                       crust_model_Gamma, crust_densities)
    crust_pressure = np.exp(integrate.cumtrapz(
        crust_Gamma, np.log(crust_densities),
        initial=0.0) + np.log(max_crust_pressure))
    crust_interal_energy = integrate.cumtrapz(
        crust_pressure/crust_densities, np.log(crust_densities), initial=0.0) + max_crust_internal_energy
    crust_energy_density = (1 + crust_interal_energy) * crust_densities
    print(len(crust_densities))
    print(len(crust_pressure))
    print(len(crust_energy_density))
    return pd.DataFrame({
        "baryon_density": np.concatenate(
            [crust_densities[::-1], eos["baryon_density"]]),
        "pressurec2": np.concatenate(
            [crust_pressure[::-1], eos["pressurec2"]]),
        "energy_densityc2": np.concatenate(
            [crust_energy_density[::-1], eos["energy_densityc2"]])})
def univ_stitch_crust_to_eos(eos):
    phi = np.log(np.gradient(eos["energy_densityc2"], eos["pressurec2"])-1)
    u_eos.set_crust(crust_eos="./ingo-bps-with-cs2c2.csv")
    eos, cols = integrate_phi(
        np.array(eos["pressurec2"]),
        np.array(phi),
        1e11,
        sigma_logpressurec2=0.0,
        stitch_below_reference_pressure=True,
        include_baryon_density=True,
        include_cs2c2=True,
        include_baryon_chemical_potential=False,
        verbose=False,
    )
    return pd.DataFrame(data=eos, columns=cols)
if __name__ == "__main__":
    eos = pd.read_csv("rmf-1109-000338.csv")
    crust = pd.read_csv("./ingo-bps-with-cs2c2.csv")
    stitched_eos = univ_stitch_crust_to_eos(eos)
    print(stitched_eos)
    stitched_eos.to_csv("rmf-1109-000338-bps_univ.csv", index=False)
    

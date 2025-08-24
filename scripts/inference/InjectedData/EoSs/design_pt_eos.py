import pandas as pd
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate

import matplotlib.pyplot as plt
rhonuc = 2.8e14

def transition_to_css_new(eos, e_onset, cs2_extension, delta_e_over_e):
    """
    An EoS which transitions from the eos given to a css EoS at the energy density 
    e_onset.  The phase transition has a latent heat such that the total energy density
    at the end of the transition over the onset is delta_e_over_e
    """
    
    e_orig = eos["energy_densityc2"]
    p_orig = eos["pressurec2"]
    rho_orig = eos["baryon_density"]

    p_onset = interpolate.griddata(e_orig, p_orig, e_onset)
    rho_onset= interpolate.griddata(e_orig, rho_orig, e_onset)
    h_onset = (p_onset + e_onset) / rho_onset
    e_extension = np.linspace(e_onset, max(e_orig), 1000)
    cs2_modified = np.where(e_extension > e_onset * delta_e_over_e, cs2_extension, 0.0)
    p_extension = integrate.cumulative_trapezoid( cs2_modified,e_extension, initial=0) + p_onset
    h_extension = np.exp(integrate.cumulative_trapezoid(cs2_modified / (e_extension + p_extension), e_extension, initial=0))  * h_onset 
    rho_extension = (e_extension + p_extension) / h_onset
    rho_entire = np.concatenate([rho_orig[e_orig<e_onset], rho_extension])
    p_entire = np.concatenate([p_orig[e_orig < e_onset], p_extension])
    e_entire = np.concatenate([e_orig[e_orig < e_onset], e_extension])
    return pd.DataFrame({"baryon_density" : rho_entire, "pressurec2":p_entire,
    "energy_densityc2" : e_entire})
def transition_to_css(eos, e_trans = 2*rhonuc, cs2_extension=.7, delta_e_over_e=1.3):
    e_orig = eos["energy_densityc2"]
    p_orig = eos["pressurec2"]
    rho_orig = eos["baryon_density"]

    cs2_orig = np.gradient(p_orig, e_orig)
    # Small number to avoid zero sound speed problems
    cs2_modified = np.where(e_orig < e_trans, cs2_orig, 1e-10)
    cs2_modified = np.where(e_orig > e_trans * delta_e_over_e, cs2_extension, cs2_modified)
    p_new = integrate.cumulative_trapezoid(cs2_modified, e_orig, initial=0.0) + p_orig[0]
    p_new = np.where(e_orig < e_trans, p_orig, p_new)
    # Computing rho precisely requires some very ridiculous tricks
    e_onset = e_trans
    p_onset = interpolate.griddata(e_orig, p_orig, e_trans)
    rho_onset= interpolate.griddata(e_orig, rho_orig, e_trans)
    p_computable = np.geomspace(p_onset, max(p_new), 1000)
    e_computable = interpolate.griddata(p_new, e_orig, p_computable)
    dlnh_dp  = interpolate.griddata(p_new, 1/(e_orig + p_new), p_computable)
    h_computable = np.exp(integrate.cumulative_trapezoid(dlnh_dp, p_computable, initial=0.0)) * (e_onset+ p_onset)/rho_onset
    rho_computed = (e_computable + p_computable)/h_computable
    print("pnew is ", p_new[e_orig > e_trans]/2.8e14)
    print("p onset is", p_onset/2.8e14)
    print("rho_computed is", rho_computed/2.8e14)
    print("rho_onset is", rho_onset/2.8e14)
    # Interpolation breaks because p_computable doesn't extend low enough? 
    rho_extension = interpolate.griddata(p_computable, rho_computed, p_new[e_orig > e_trans])
    print("rho_extension", rho_extension/2.8e14)
    rho_new = np.concatenate([rho_orig[e_orig < e_trans], rho_extension])
    return pd.DataFrame({"energy_densityc2": e_orig, "pressurec2":p_new, "baryon_density":rho_new})
 
if __name__ == "__main__":
    eos_to_modify = pd.read_csv("./rmf-1109-000338-bps.csv")
    new_eos = transition_to_css_new(eos_to_modify, e_onset = 2.0*rhonuc, cs2_extension=0.7, delta_e_over_e=1.3)
    print(new_eos["pressurec2"]/ eos_to_modify["pressurec2"])
    print(new_eos)
    new_eos.to_csv("rmf-1109-000338-bps_2p0_1p3_0p7.csv", index=False)

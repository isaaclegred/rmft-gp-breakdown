"""
Get a file to store EoS properties for each EoS set, including likelihoods. 
"""
import temperance as tmpy
import temperance.sampling.eos_prior as tmeos_prior

import pandas as pd
import numpy as np
import tqdm
import os

def get_mmax(macro_data):
    dM_drho = np.gradient(macro_data["M"], macro_data["central_baryon_density"])
    stable = np.where(dM_drho > 0)[0]
    return np.max(macro_data["M"][stable])
    


def get_eos_file(eos_directory, properties = {"Mmax": get_mmax}, num_eoss=3000, eos_per_dir=1000):
    """
    For each eos, get all of the properties specified in the properties (currently only can get
    properties from the "macro" file)
    """
    eos_prior_set = tmeos_prior.EoSPriorSet(eos_dir = eos_directory,eos_column= "eos", eos_per_dir=eos_per_dir, macro_dir=eos_directory, branches_data=None)
    eos_file = pd.DataFrame({"eos" : np.arange(num_eoss) })
    for prop_name in properties.keys():
        eos_file[prop_name] = np.zeros(num_eoss)
    for eos_index in tqdm.trange(num_eoss):
        macro_data = pd.read_csv(eos_prior_set.get_macro_path(eos_index))
        for prop_name in properties.keys():
            eos_file.loc[eos_index, prop_name] = properties[prop_name](macro_data)
    return eos_file

if __name__ == "__main__":
    for directory in [
            "11233_maxpc2-1e11_mrgagn_01d000_00d010",
            "11233_maxpc2-1e12_mrgagn_01d000_00d010",
            "11233_maxpc2-1e13_mrgagn_01d000_00d010",
            "11233_maxpc2-1e14_mrgagn_01d000_00d010"
    ]:
        eos_file = get_eos_file(os.path.join("/home/isaac.legred/RMFGP/gp-rmf-inference/conditioned-priors/11233", 
                                directory))
        eos_file.to_csv(f"{directory}.csv", index=False)
            

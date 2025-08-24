import numpy as np
import bilby
import json
import glob
import pandas as pd
from tqdm import trange

max_f = 2048
min_f = 23

def get_bns_prior(chirp_mass):
    prior_template = bilby.gw.prior.BNSPriorDict(aligned_spin=True)
    prior_template['chirp_mass'] = bilby.core.prior.Uniform(name = 'chirp_mass', minimum = chirp_mass - 0.01, maximum= chirp_mass + 0.01)
    prior_template['mass_ratio'] = bilby.core.prior.Uniform(name = 'mass_ratio', minimum = 0.125, maximum = 1)
    for ii in [1,2]:
        del prior_template[f"mass_{ii}"]
        prior_template[f"chi_{ii}"].a_prior.maximum = 0.05
        prior_template[f"lambda_{ii}"].maximum = 15000
    prior_template['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(minimum = 1, maximum = 1500, name = 'luminosity_distance', latex_label='$d_L$')
    
    new_prior = {key: prior_template[key] for key in prior_template.keys()}
    return new_prior

def get_duration(sample, fmin):    
    duration = bilby.gw.utils.calculate_time_to_merger(fmin, sample['mass_1'], 
                sample['mass_2'])
    duration = max(4, 2**(np.ceil(np.log2(duration))))
    return duration

def generate_ini(params, name, fmax, fmin, subdir):
    with open('template.ini', 'r') as rr:
        template = rr.readlines()

    params['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(mass_1=params['mass_1'], mass_2 = params['mass_2'])
    
    duration = get_duration(params, min_f)
  
    prior_here = get_bns_prior(params['chirp_mass'])
    #prior_here['luminosity_distance'].cosmology = None
    #prior_here['luminosity_distance'].unit = None
    for jj, line in enumerate(template):
        if line.startswith("duration="):
            template[jj] = f"duration={duration}\n"
        if line.startswith("sampling-frequency="):
            template[jj] = f"sampling-frequency={fmax * 2}\n"
        if line.startswith("maximum-frequency="):
            template[jj] = f"maximum-frequency={fmax * 7/8}\n"
        if line.startswith("injection-dict="):
            template[jj] = f"injection_dict={params}\n"
        if line.startswith("label="):
            template[jj] = f"label={name}\n"
        if line.startswith("prior-dict="):
            template[jj] = "prior-dict={"
            for key in prior_here.keys():
                template[jj] += f"{key}:{prior_here[key]},"
            template[jj] = template[jj].strip(',')
            template[jj] += "}\n"
        if line.startswith("outdir="):
            template[jj] = f"outdir={subdir}/outdir_{name}\n"
    return template
def get_and_write_ini(indx, df, fmax, fmin, subdir):
    params_here = {key: df.loc[key][0][indx] for key in df.index if "Unnamed" not in key}
    print(params_here)
    name_here = f"injection_{indx}"
    ini = generate_ini(params_here, name_here, fmax, fmin, subdir)
    with open(f"{subdir}/{name_here}.ini", "w") as ww:
        ww.writelines(ini)

if __name__ == "__main__":
    eos_dir = "rmf-1109-000338-bps_2p0_1p3_0p7"
    eos_label = eos_dir
    events = pd.read_json(f"{eos_dir}/found_injections-{eos_label}.json")

    for i in trange(len(events.iloc[0]["injections"])):
        get_and_write_ini(i, events, max_f, min_f, subdir=f"{eos_dir}")

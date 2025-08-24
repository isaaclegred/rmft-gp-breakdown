import os
import glob
import h5py
import pandas as pd
import temperance as tmpy
import temperance.sampling.eos_prior as eos_prior
from temperance.core import result
from temperance.sampling.eos_prior import EoSPriorSet
from temperance.core.result import EoSPosterior
import numpy as np

import tqdm as tqdm



# === User configuration ===

output_file = "ns_data.h5"              # Output HDF5 filename


# === Script ===
def csvs_to_h5( indices, output_file, eos_pattern=lambda x : f"eos_{x}.csv", macro_pattern=lambda x : f"macro_{x}.csv", weights=None):


    with h5py.File(output_file, "w") as h5f:
        eos_group = h5f.create_group("EoS")
        macro_group = h5f.create_group("Macro")
        weights_group = h5f.create_group("Weights")


        for idx in tqdm.tqdm(indices):
            # Load CSVs into pandas DataFrames
            eos_file = eos_pattern(idx)
            macro_file = macro_pattern(idx)
            eos_df = pd.read_csv(eos_file)
            macro_df = pd.read_csv(macro_file)
            if weights is not None:
                weights_group.create_dataset(str(idx), data=weights[idx])

            # Convert to numpy arrays for HDF5 storage
            eos_data = eos_df.to_numpy()
            macro_data = macro_df.to_numpy()

            # Store in HDF5 with the index as the key
            eos_group.create_dataset(str(idx), data=eos_data)
            macro_group.create_dataset(str(idx), data=macro_data)

            # # Optional: store column headers as attributes
            # eos_group[str(idx)].attrs["columns"] = list(eos_df.columns)
            # macro_group[str(idx)].attrs["columns"] = list(macro_df.columns)

            # # Also store original file names as attributes for reference
            # eos_group[str(idx)].attrs["source_file"] = os.path.basename(eos_file)
            # macro_group[str(idx)].attrs["source_file"] = os.path.basename(macro_file)
        

    print(f"HDF5 file '{output_file}' created with {len(indices)} EoS and macroscopic datasets.")


if __name__ == "__main__":
    rmf_eos_set = "1109"
    main_dir = "/home/isaac.legred/RMFGP"
    for key in ["1e11", "1e12", "3e12", "1e13", "3e13", "1e14"]:
        eos_set_identifier=f"maxpc2-{key}"
        eos_tag = f"{rmf_eos_set}_{eos_set_identifier}_mrgagn_01d000_00d010"
        rmf_dir  = f"{main_dir}/gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/{eos_tag}"
        rmf_eos_post = EoSPosterior.from_csv(
            f"{main_dir}/RealData/{eos_tag}_post.csv", label="astro" + "-" + eos_set_identifier + f"-{rmf_eos_set}")
        rmf_eos_prior = EoSPriorSet(
            eos_dir=rmf_dir, eos_column="eos", eos_per_dir=1000, macro_dir=rmf_dir,
            branches_data=None)
        rmf_weight_columns = [
            result.WeightColumn(column_name, is_log=True) for column_name in
            [
                'logweight_Antoniadis_J0348',
                'logweight_Choudhury_J0437',
                'logweight_Miller_J0740',
                'logweight_Miller_J0030',
                'logweight_gw_170817',
                'logweight_gw_190425']
        ]
        transition_pressure = eval(key)
        exponent = int(np.log10(transition_pressure))
        # Optional: pattern to match files (can be sorted for consistent indexing)
        eos_pattern = lambda eos_index : rmf_eos_prior.get_eos_path(eos_index)
        macro_pattern = lambda eos_index : rmf_eos_prior.get_macro_path(eos_index)
        indices = np.array(rmf_eos_post.samples["eos"])
        output_file = f"./{eos_tag}_ns_data.h5"
        # Create another dataset in the h5 file which contains the weights for each EoS
        weights = rmf_eos_post.get_total_weight(rmf_weight_columns)["total_weight"]

        csvs_to_h5(indices, output_file, eos_pattern, macro_pattern, weights=weights)
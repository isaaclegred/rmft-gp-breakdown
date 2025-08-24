import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bilby
if __name__=="__main__":
    save_injections=True
    eos = "rmf-1109-000338-bps_2p0_1p3_0p7"
    with open(f"samples_low_mass_nearby-macro-{eos}_0.json") as f:
        data = json.loads(f.read())
    plt.hist(np.array(data["samples"]["luminosity_distance"])[data["samples"]["all_cuts_passed"]], bins=30)
    print("found", len(np.array(data["samples"]["luminosity_distance"])[data["samples"]["all_cuts_passed"]]), "injections out of")
    print(len(np.array(data["samples"]["luminosity_distance"])))
    plt.xlabel("dL")
    if save_injections:
        with open(f"found_injections-{eos}.json", "w+") as f:
            samples_to_save = pd.DataFrame(data["samples"])
            samples_to_save = samples_to_save[samples_to_save["all_cuts_passed"]==1.0]
            data_w_redshift = bilby.gw.conversion.generate_source_frame_parameters(pd.DataFrame(samples_to_save))
            data_w_redshift["mass_1"] = data_w_redshift["mass_1_source"] * (1 + data_w_redshift["redshift"])
            data_w_redshift["mass_2"] = data_w_redshift["mass_2_source"] * (1 + data_w_redshift["redshift"])
            data_w_redshift["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(data_w_redshift["mass_1"], data_w_redshift["mass_2"])
            json.dump({"injections": data_w_redshift.to_dict("list")}, f)
    plt.savefig(f"samples_low_mass_nearby-macro-{eos}.pdf")

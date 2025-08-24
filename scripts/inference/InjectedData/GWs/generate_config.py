import configparser
import os
import pandas as pd
import numpy as np
config = configparser.ConfigParser()
def generate_config(eos_directory,
                    eos_per_dir, eos_tag,
                    eos_indices_file, gw_posterior_samples,
                    gw_posterior_in_file, outdir,  outtag):
  in_data = pd.read_csv(gw_posterior_in_file, index_col=0)
  config['Samples'] = {'eos-directory': eos_directory,
                       'eos-per-dir': eos_per_dir,
                       'eos-indices': eos_indices_file,
                       'gw-posterior-samples':
                       gw_posterior_samples,
                       'bandwidth-file': gw_posterior_in_file}
  config['Marginalization'] = {'prior' : 'default',
                               'chirp-mass-range' : str((.99 * in_data.loc["mc", "lb"],
                                                         1.01 * in_data.loc["mc", "ub"])),
                               'mass-ratio-range' :str((.99 * in_data.loc["q", "lb"],
                                                        min(1.01 * in_data.loc["q", "ub"], 1.0
                                                        )))
  }
  config['Submission'] = {'label' : f'{outtag}',
                          'format-for-condor': 'True',
                          'condor-num-jobs': '10',
                          'accounting' : 'ligo.sim.o4.cbc.extremematter.bilby',
                          'submit-dag' : 'True',
                          'merge-executable' : '/home/isaac.legred/lwp/bin/combine-samples',
                          'condor-kwargs' : {'request_disk' : '10 MB',
                                             'request_memory': '2048 MB',
                                             'universe' : 'vanilla'
                          }
  }
  config["Output"] = {'save-marginalized-likelihoods' : f'{outtag}_eos.csv',
                      'save-likelihoods': f'{outtag}_post.csv',
                      'output-dir': os.path.join(outdir, outtag)}
  
  with open(f'{outdir}/{outtag}.ini', 'w') as configfile:
    config.write(configfile)

    
  # with open('example.ini', 'r') as configfile:
  #   config.read(configfile)
def generate_all_lwp_inis(eos_tag, eos_directory, eos_indices_file, eos_per_dir, injection_dir, outdir, gw_tags):
  for gw_tag in gw_tags:
    outtag = gw_tag
    gw_posterior_samples = f"{injection_dir}/{gw_tag}_result.csv"
    gw_posterior_in_file = f"{injection_dir}/{gw_tag}_result.in"
    generate_config(
      eos_directory,
      eos_per_dir, eos_tag,
      eos_indices_file, gw_posterior_samples,
      gw_posterior_in_file, outdir,
      outtag)

  
    
if __name__ == "__main__":
    make_manifest = False
    rmf_set_used = "1109"

    for eos_prior_set in [
          "1109_maxpc2-1e11_mrgagn_01d000_00d010",
          "1109_maxpc2-1e12_mrgagn_01d000_00d010",
          "1109_maxpc2-3e12_mrgagn_01d000_00d010",
          "1109_maxpc2-1e13_mrgagn_01d000_00d010",
          "1109_maxpc2-3e13_mrgagn_01d000_00d010",
          "1109_maxpc2-1e14_mrgagn_01d000_00d010",
        "rmf_1109"
    ]:
      eos_tag = eos_prior_set
      eos_directory = f"/home/isaac.legred/RMFGP/gp-rmf-inference/conditioned-priors/many-draws/{rmf_set_used}/{eos_tag}"
      eos_per_dir = 1000
      injection_dir=f"{os.getcwd()}/rmf-1109-000338-bps_2p0_1p3_0p7"
      eos_indices_file = f"{eos_directory}/manifest.csv"

      if not os.path.exists(injection_dir +"/"+ eos_prior_set):
        os.mkdir(injection_dir +"/"+ eos_prior_set)

      if make_manifest:
        pd.DataFrame({"eos": np.arange(30000)}).to_csv(
          eos_indices_file, index=False)

      runs_to_do = np.array([0, 1, 2, 3,5])
      
      #runs_to_do = np.delete(runs_to_do, 15)
      #runs_to_do = np.delete(runs_to_do, 20)
      #runs_to_do = np.delete(runs_to_do, 0)
      
      gw_tags = [f"injection_{i}" for i in runs_to_do]
      generate_all_lwp_inis(eos_tag, eos_directory,  eos_indices_file,
                            eos_per_dir, injection_dir,
                            injection_dir + "/" + eos_prior_set, gw_tags)
  

Gravitational wave injections simulated gaussian noise based on A plus sensitivity.
We sample from a population outlined in bilby format in sampling.prior, and take perform
parameter estimation on the injections using bilby, and then modify the tidal deformabilities
of the injections to agree with a chosen EoS.
We inject and recover using IMR_PhenomPv2_NRTidalv2.

We then perform hierarchical inference using lwp.  We infer using each of the injection sets.



./sample_population.py -> Sample the GW population
./diagnose_samples.py -> Format samples which pass an SNR cut for use in bilb
./generate_inis.ipynb -> jupyter notebook to modify bilby template ini file (template.ini)
bilby inis are then launched (most easily by a directory dependent helper script)
./get_in_file.py -> find the bandwidth for the KDE for LWP for each bilby posterior
./generate_config.py -> generate the lwp config files
lwp inis are then launched for each EoS set, EoS likelihoods are computed.



Code for inferring the EoS using RMF conditioned GPs

We use 3 different data sources

NICER/XMM -> ./XRAY
Radio pulsar timing -> ./RADIO
Gravitational Waves -> ./GWs

./get_eos_files.py:
Collate the eos indices and their maximum masses.

./merge_likelihoods.py -> combine all likelihoods from different mock data into a single csv file
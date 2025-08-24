EoS inference using injected radio sources.

We choose masses near Mtov for an injected equation of state, assign a random uncertainty, and
then draw samples from a gaussian distribution representing the measurement.
./make_fake_psr_samples.py

We then analyze sourcesunder the assumption that every observation informs Mtov, i.e.
that we need to consider an Occam factor in the analysis.
./weigh_eos_set_by_pulsar.py




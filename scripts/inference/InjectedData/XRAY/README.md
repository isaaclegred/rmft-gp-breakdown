Inference on injected X-ray (NICER/XMM) Mass-radius posteriors.

We first sample "true-value"
mass-radii according to some equation of state (currently we hardcode masses
because we have no good model for NICER selection).  We assign random uncertainty to the mass and
the radius based on some (potentially unreasonable) models (we should likely discuss this).

See ./make_fake_nicer_samples.py

We then perform the NICER inference by sampling masses and equations of state, we draw samples
from a fixed mass-population based on the source type (in analogy with the current astrophysical case).  If the source would inform Mtov, then we sample from 1 solar mass up to Mtov, which produces
an occam factor for EoSs which predict the existence of very massive stars.  If the mass sampled
would not inform Mtov meaningfully, we assume a flat mass population prior.  This is something we could discuss, and represents an analysis setting, rather than a fixed assumption.

See ./analyze_nicer.py


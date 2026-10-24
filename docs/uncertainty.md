# Correlated uncertainty

linearized_covariance evaluates J Sigma J^T after checking dimensions, symmetry
and positive semidefiniteness. Correlation terms must be retained when geometry
parameters share manufacturing errors.

The coax study assumes normal errors in inner radius, outer radius and relative
permittivity. Their standard deviations and the radius correlation are explicit
parameters. Named deterministic random streams keep the sample identity stable
when unrelated study code adds new stochastic draws.

Every draw is checked for positive radius, ordered conductors and positive
permittivity. The model does not silently clip impossible geometry into validity.

## Two distinct intervals

The empirical 2.5th and 97.5th capacitance percentiles describe unit-to-unit
population spread under the assumed tolerances.

mean_bootstrap resamples independent complete units and returns a percentile
confidence interval for the population mean. It becomes narrower as the number
of independent simulated units increases; that does not improve an individual
manufactured unit's tolerance.

The reported sample standard deviation is compared with the linearized derivative
prediction. The histogram is a density, so its ordinate carries inverse-capacitance
units. These samples are assumed-model simulations rather than measured production
statistics. Model uncertainty and systematic calibration errors are not estimated.

Correlated time samples, shared batches or hierarchical manufacturing data require
a corresponding resampling model; the iid mean bootstrap is inappropriate for them.

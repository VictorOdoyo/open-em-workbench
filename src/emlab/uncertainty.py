import numpy as np


def linearized_covariance(jacobian, parameter_covariance):
    from .core import finite_array

    jacobian = finite_array(jacobian, real=True)
    covariance = finite_array(parameter_covariance, real=True, ndim=2)
    if jacobian.ndim == 1:
        jacobian = jacobian[None, :]
    scale = max(float(np.max(abs(covariance))), np.finfo(float).tiny)
    if (
        jacobian.ndim != 2
        or covariance.shape != (jacobian.shape[1], jacobian.shape[1])
        or not np.allclose(covariance, covariance.T, atol=scale * 1e-12)
        or np.min(np.linalg.eigvalsh(covariance)) < -scale * 1e-12
    ):
        raise ValueError("matching positive-semidefinite parameter covariance required")
    return jacobian @ covariance @ jacobian.T


def mean_bootstrap(samples, confidence=0.95, resamples=2000, seed=42):
    from .core import finite_array, random_stream

    samples = finite_array(samples, real=True, ndim=1)
    if len(samples) < 2 or not 0 < confidence < 1 or not isinstance(resamples, int) or resamples < 100:
        raise ValueError("independent samples confidence in (0,1) and at least 100 resamples required")
    rng = random_stream(seed, "mean-bootstrap")
    indices = rng.integers(0, len(samples), size=(resamples, len(samples)))
    means = np.mean(samples[indices], axis=1)
    tail = (1 - confidence) / 2
    return tuple(float(v) for v in np.quantile(means, [tail, 1 - tail]))

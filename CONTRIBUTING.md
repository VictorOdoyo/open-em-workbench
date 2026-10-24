# Contributing

Use Python 3.11 or newer. Install the locked developer environment with
uv sync --locked --extra dev. Run uv run pytest -q, uv run ruff check src tests examples,
and uv run ruff format --check src tests examples before submitting a change.

For a numerical feature, provide an independent analytical limit, conservation law,
manufactured solution, or separately derived recurrence. A test that repeats the
implementation's formula does not establish physical validity. Record units, phasor
signs, field staggering, boundary assumptions and the supported regime.

Keep changes focused and commits substantive. Do not add empty history or synthetic
timestamps. Numerical tolerances should reflect the stated grid/truncation error
and work on Linux and Windows, rather than hiding a convergence defect.

A new catalog case must return a validated result with explicit axes and physical
parameters, register in experiments.CASES, include a targeted physics test and update
the method/case documentation. Preserve honest nonmonotonic geometry effects.

Published results must come from one clean committed source revision. Capture into
ignored outputs/, render and verify, then use examples/build_gallery.py. Never edit
an old provenance record to claim a newer calculation. Inspect generated figures
for masks, units, aliasing, coordinate ranges and unreadable labels.

Archives and manifests use LF text. Do not commit credentials, virtual environments,
scratch outputs or raw external solver files. Third-party methods need primary
references and independently written code or explicit compatible attribution.

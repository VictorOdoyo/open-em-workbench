from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid2D:
    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self):
        if any(not isinstance(n, int) or isinstance(n, bool) or n < 3 for n in (self.nx, self.ny)):
            raise ValueError("at least three nodes per Cartesian axis required")
        if not np.all(np.isfinite([self.dx, self.dy, self.x0, self.y0])) or min(self.dx, self.dy) <= 0:
            raise ValueError("positive finite spacing and finite origin required")

    @property
    def shape(self):
        return self.ny, self.nx

    @property
    def x(self):
        return self.x0 + np.arange(self.nx) * self.dx

    @property
    def y(self):
        return self.y0 + np.arange(self.ny) * self.dy

    def mesh(self):
        return np.meshgrid(self.x, self.y, indexing="xy")

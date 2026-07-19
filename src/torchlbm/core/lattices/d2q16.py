"""D2Q16 = D1Q4 (x) D1Q4 Gauss-Hermite lattice for PonD (lattice temperature T_L = 1).

D1Q4 is exact to the 7th Gaussian moment (<c^6> = 15), which D1Q3 misses -- that is
what buys high-Mach Galilean invariance. Velocities are stored in tensor order
t = 4*kx + ky, so the gauge transform factorises as Gx (kron) Gy with no permutation.
The abscissae are irrational (not space-filling), which is fine: PonD advects semi-
Lagrangian, off-grid.
"""
import math
from typing import List

from .lattice import Lattice

# D1Q4 Gauss-Hermite abscissae (+-inner, +-outer) and their weights.
_INNER = math.sqrt(3.0 - math.sqrt(6.0))
_OUTER = math.sqrt(3.0 + math.sqrt(6.0))
_W_INNER = (3.0 + math.sqrt(6.0)) / 12.0
_W_OUTER = (3.0 - math.sqrt(6.0)) / 12.0

D1Q4_ABSCISSAE: List[float] = [-_OUTER, -_INNER, _INNER, _OUTER]
D1Q4_WEIGHTS: List[float] = [_W_OUTER, _W_INNER, _W_INNER, _W_OUTER]

D2Q16_LATTICE_TEMPERATURE = 1.0


class D2Q16(Lattice):
    """D2Q16 Gauss-Hermite velocity set in tensor order t = 4*kx + ky."""

    def __init__(self) -> None:
        self.n_discrete_velocities = 16
        self.lattice_temperature = D2Q16_LATTICE_TEMPERATURE

        # Outer product of the two 1-D sets: t = 4*kx + ky, weight = w1[kx]*w1[ky].
        c1 = D1Q4_ABSCISSAE
        w1 = D1Q4_WEIGHTS
        cx, cy, cz, w = [], [], [], []
        for kx in range(4):
            for ky in range(4):
                cx.append(c1[kx])
                cy.append(c1[ky])
                cz.append(0.0)
                w.append(w1[kx] * w1[ky])
        self.my_lattice_velocities = [cx, cy, cz]
        self.my_lattice_weights = w
        self.my_lattice_indices = list(range(16))
        # Opposite of (kx, ky) is (3-kx, 3-ky) since c1[3-k] = -c1[k].
        self.my_opposite_lattice_indices = [
            4 * (3 - (t // 4)) + (3 - (t % 4)) for t in range(16)
        ]
        # Half-space index groups by component sign (used by wall / inflow BCs).
        self.my_east_velocities = [t for t in range(16) if cx[t] > 0]
        self.my_west_velocities = [t for t in range(16) if cx[t] < 0]
        self.my_north_velocities = [t for t in range(16) if cy[t] > 0]
        self.my_south_velocities = [t for t in range(16) if cy[t] < 0]
        self.my_top_velocities: List[int] = []
        self.my_bottom_velocities: List[int] = []

    def number_of_discrete_velocities(self) -> int:
        return self.n_discrete_velocities

    def lattice_velocities(self) -> List[List[float]]:
        return self.my_lattice_velocities

    def lattice_indices(self) -> List[int]:
        return self.my_lattice_indices

    def opposite_lattice_indices(self) -> List[int]:
        return self.my_opposite_lattice_indices

    def lattice_weights(self) -> List[float]:
        return self.my_lattice_weights

    def population_to_momentum_transform(self) -> List[List[float]]:
        # MRT transforms are D2Q9-specific; PonD never uses them on D2Q16.
        raise NotImplementedError("D2Q16 has no MRT population<->momentum transform (PonD does not use it).")

    def momentum_to_population_transform(self) -> List[List[float]]:
        raise NotImplementedError("D2Q16 has no MRT population<->momentum transform (PonD does not use it).")

    def east_velocities(self) -> List[int]:
        return self.my_east_velocities

    def west_velocities(self) -> List[int]:
        return self.my_west_velocities

    def north_velocities(self) -> List[int]:
        return self.my_north_velocities

    def south_velocities(self) -> List[int]:
        return self.my_south_velocities

    def top_velocities(self) -> List[int]:
        return self.my_top_velocities

    def bottom_velocities(self) -> List[int]:
        return self.my_bottom_velocities

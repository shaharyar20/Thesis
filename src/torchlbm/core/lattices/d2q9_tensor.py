"""D2Q9 = D1Q3 (x) D1Q3 in **tensor order** ``t = 3*kx + ky`` (for the PonD solver).

Physically identical to :class:`torchlbm.core.lattices.d2q9.D2Q9` (same abscissae
``c1 = {-1, 0, 1}``, same weights, ``T_L = 1/3``) but laid out in tensor order so the
general n-point gauge transform (:class:`PondGaugeTransform`) applies as a plain
``Gx (kron) Gy`` with no permutation. The standard-order ``D2Q9`` is kept untouched for
the (non-PonD) exponential-equilibrium solver, which relies on its MRT transforms.

This is the D2Q9 *fallback* path for the moving-frame / jet drivers (ambient at rest,
open at any Mach); the primary high-speed path is D2Q16.
"""

from typing import List

from .lattice import Lattice

# D1Q3 abscissae (increasing) and weights; T_L = sum w c^2 = 1/3.
D1Q3_ABSCISSAE: List[float] = [-1.0, 0.0, 1.0]
D1Q3_WEIGHTS: List[float] = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]

D2Q9_TENSOR_LATTICE_TEMPERATURE = 1.0 / 3.0


class D2Q9Tensor(Lattice):
    """D2Q9 velocity set in tensor order ``t = 3*kx + ky`` (PonD fallback lattice)."""

    def __init__(self) -> None:
        self.n_discrete_velocities = 9
        self.lattice_temperature = D2Q9_TENSOR_LATTICE_TEMPERATURE

        c1 = D1Q3_ABSCISSAE
        w1 = D1Q3_WEIGHTS
        cx, cy, cz, w = [], [], [], []
        for kx in range(3):
            for ky in range(3):  # t = 3*kx + ky
                cx.append(c1[kx])
                cy.append(c1[ky])
                cz.append(0.0)
                w.append(w1[kx] * w1[ky])
        self.my_lattice_velocities = [cx, cy, cz]
        self.my_lattice_weights = w
        self.my_lattice_indices = list(range(9))
        # opposite of (kx, ky) is (2-kx, 2-ky) since c1[2-k] = -c1[k].
        self.my_opposite_lattice_indices = [
            3 * (2 - (t // 3)) + (2 - (t % 3)) for t in range(9)
        ]
        self.my_east_velocities = [t for t in range(9) if cx[t] > 0]
        self.my_west_velocities = [t for t in range(9) if cx[t] < 0]
        self.my_north_velocities = [t for t in range(9) if cy[t] > 0]
        self.my_south_velocities = [t for t in range(9) if cy[t] < 0]
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
        raise NotImplementedError("D2Q9Tensor has no MRT transform (PonD does not use it).")

    def momentum_to_population_transform(self) -> List[List[float]]:
        raise NotImplementedError("D2Q9Tensor has no MRT transform (PonD does not use it).")

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

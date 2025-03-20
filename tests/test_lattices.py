import pytest
from torchlbm.core.lattices.lattice_dictionaries import (
    OneDimensionalLattices,
    TwoDimensionalLattices,
    ThreeDimensionalLattices,
)


def test_lattices():
    def test_lattice_dict(lattice_dictionary):
        for key, value in lattice_dictionary.items():
            assert sum(value.lattice_weights()) == pytest.approx(1.0)
            assert sum([sum(dim) for dim in value.lattice_velocities()]) == pytest.approx(0.0)

    test_lattice_dict(OneDimensionalLattices)
    test_lattice_dict(TwoDimensionalLattices)
    test_lattice_dict(ThreeDimensionalLattices)

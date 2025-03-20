from typing import List

from torchlbm.exceptions import TorchlbmError
from .lattice import Lattice


class D1Q2(Lattice):
    """The velocity set for the D1Q2 velocity set.

    Args:
        Lattice (Lattice): The base class prescribing methods the respective velocity set should implement.
    """

    def __init__(self) -> None:
        """Initializer for the D1Q2 velocity set."""
        self.n_discrete_velocities = 2

        self.my_lattice_velocities = [
            [1.0, -1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]

        self.my_lattice_indices = [
            0,
            1,
        ]

        self.my_opposite_lattice_indices = [
            1,
            0,
        ]

        self.my_lattice_weights = [
            1.0 / 2.0,
            1.0 / 2.0,
        ]

        self.my_east_velocities = None
        self.my_west_velocities = None
        self.my_north_velocities = None
        self.my_south_velocities = None
        self.my_top_velocities: List[int] = []
        self.my_bottom_velocities: List[int] = []
        self.my_pure_vertical_velocities = None
        self.my_pure_horizontal_velocities = None

    def number_of_discrete_velocities(self) -> int:
        """See base class.

        Returns:
            int: See base class.
        """
        return self.n_discrete_velocities

    def lattice_velocities(self) -> List[List[float]]:
        """See base class.

        Returns:
            List[List[float]]: See base class.
        """
        return self.my_lattice_velocities

    def lattice_indices(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_lattice_indices

    def opposite_lattice_indices(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_opposite_lattice_indices

    def lattice_weights(self) -> List[float]:
        """See base class.

        Returns:
            List[float]: See base class.
        """
        return self.my_lattice_weights

    def population_to_momentum_transform(self) -> List[List[float]]:
        """See base class.

        Returns:
            List[List[float]]: See base class.
        """
        raise TorchlbmError("The method population_to_momentum_transform in the D1Q2 velocity is not yet implemented.")

    def momentum_to_population_transform(self) -> List[List[float]]:
        """See base class.

        Returns:
            List[List[float]]: See base class.
        """
        raise TorchlbmError("The method momentum_to_population_transform in the D1Q2 velocity is not yet implemented.")

    def east_velocities(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_east_velocities

    def west_velocities(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_west_velocities

    def north_velocities(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_north_velocities

    def south_velocities(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_south_velocities

    def top_velocities(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_top_velocities

    def bottom_velocities(self) -> List[int]:
        """See base class.

        Returns:
            List[int]: See base class.
        """
        return self.my_bottom_velocities

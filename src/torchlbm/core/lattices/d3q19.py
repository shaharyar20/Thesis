from typing import List

from torchlbm.exceptions import TorchlbmError
from .lattice import Lattice


class D3Q19(Lattice):
    """The velocity set for the D3Q19 velocity set.

    Args:
        Lattice (Lattice): The base class prescribing methods the respective velocity set should implement.
    """

    def __init__(self) -> None:
        """Initializer for the D3Q19 velocity set."""
        self.n_discrete_velocities = 19

        self.my_lattice_velocities = [
            [0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        ]  #  0    1    2    3    4    5    6     7    8    9    10   11    12   13   14   15    16   17    18
        self.my_lattice_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.my_opposite_lattice_indices = [0, 2, 1, 4, 3, 6, 5, 12, 11, 14, 13, 8, 7, 10, 9, 18, 17, 16, 15]

        self.my_east_velocities: List[int] = [1, 7, 8, 9, 10]  # ToDo: Determine correct values
        self.my_west_velocities: List[int] = [2, 11, 12, 13, 14]  # ToDo: Determine correct values
        self.my_north_velocities: List[int] = [3, 7, 11, 15, 16]  # ToDo: Determine correct values
        self.my_south_velocities: List[int] = [4, 8, 12, 17, 18]  # ToDo: Determine correct values
        self.my_top_velocities: List[int] = [5, 9, 13, 15, 17]  # ToDo: Determine correct values
        self.my_bottom_velocities: List[int] = [6, 10, 14, 16, 18]  # ToDo: Determine correct values
        self.my_pure_vertical_velocities: List[int] = [0]  # ToDo: Determine correct values
        self.my_pure_horizontal_velocities: List[int] = [0]  # ToDo: Determine correct values

        self.my_lattice_weights = [
            1.0 / 3.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 18.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        ]

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

    def lattice_velocity_directions(self) -> List[List[float]]:
        """See base class.

        Returns:
            List[List[float]]: See base class.
        """
        return self.my_lattice_velocity_directions

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
        raise TorchlbmError("The method population_to_momentum_transform in the D3Q19 velocity is not yet implemented.")

    def momentum_to_population_transform(self) -> List[List[float]]:
        """See base class.

        Returns:
            List[List[float]]: See base class.
        """
        raise TorchlbmError("The method momentum_to_population_transform in the D3Q19 velocity is not yet implemented.")

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

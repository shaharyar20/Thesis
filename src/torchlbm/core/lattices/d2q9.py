from typing import List

from .lattice import Lattice


class D2Q9(Lattice):
    """The D2Q9 Velocity set to express a discretized velocity distribution function.
    It is a two-dimensional velocity set with 9 discrete velocities.
    The discrete velocities have the following directions and indices:
    6   2   5
    3   0   1
    7   4   8

    Args:
        Lattice (Lattice): The baseclass prescribing the methods a velocity set has to provide.
    """

    def __init__(self) -> None:
        """Initializer for the D2Q9 velocity set."""
        self.n_discrete_velocities = 9

        self.my_lattice_velocities = [
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]

        self.my_lattice_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]

        self.my_opposite_lattice_indices = [
            0,
            3,
            4,
            1,
            2,
            7,
            8,
            5,
            6,
        ]

        self.my_lattice_weights = [
            4.0 / 9.0,  # Center Velocity [0,]
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,  # Axis-Aligned Velocities [1, 2, 3, 4]
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,  # 45 ° Velocities [5, 6, 7, 8]
        ]

        self.my_population_to_momentum_transform = [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-4.0, -1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0],
            [4.0, -2.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0],
            [0.0, -2.0, 0.0, 2.0, 0.0, 1.0, -1.0, -1.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            [0.0, 0.0, -2.0, 0.0, 2.0, 1.0, 1.0, -1.0, -1.0],
            [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0],
        ]

        self.my_momentum_to_population_transform = [
            [1 / 9.0, -1 / 9.0, 1 / 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1 / 9.0, -1 / 36.0, -1 / 18.0, 1 / 6.0, -1 / 6.0, 0.0, 0.0, 1 / 4.0, 0.0],
            [1 / 9.0, -1 / 36.0, -1 / 18.0, 0.0, 0.0, 1 / 6.0, -1 / 6.0, -1 / 4.0, 0.0],
            [1 / 9.0, -1 / 36.0, -1 / 18.0, -1 / 6.0, 1 / 6.0, 0.0, 0.0, 1 / 4.0, 0.0],
            [1 / 9.0, -1 / 36.0, -1 / 18.0, 0.0, 0.0, -1 / 6.0, 1 / 6.0, -1 / 4.0, 0.0],
            [1 / 9.0, 1 / 18.0, 1 / 36.0, 1 / 6.0, 1 / 12.0, 1 / 6.0, 1 / 12.0, 0.0, 1 / 4.0],
            [1 / 9.0, 1 / 18.0, 1 / 36.0, -1 / 6.0, -1 / 12.0, 1 / 6.0, 1 / 12.0, 0.0, -1 / 4.0],
            [1 / 9.0, 1 / 18.0, 1 / 36.0, -1 / 6.0, -1 / 12.0, -1 / 6.0, -1 / 12.0, 0.0, 1 / 4.0],
            [1 / 9.0, 1 / 18.0, 1 / 36.0, 1 / 6.0, 1 / 12.0, -1 / 6.0, -1 / 12.0, 0.0, -1 / 4.0],
        ]

        self.my_east_velocities = [1, 5, 8]
        self.my_west_velocities = [3, 6, 7]
        self.my_north_velocities = [2, 5, 6]
        self.my_south_velocities = [4, 7, 8]
        self.my_top_velocities: List[int] = [0]
        self.my_bottom_velocities: List[int] = [0]
        self.my_pure_vertical_velocities = [0, 2, 4]
        self.my_pure_horizontal_velocities = [0, 1, 3]

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
        return self.my_population_to_momentum_transform

    def momentum_to_population_transform(self) -> List[List[float]]:
        """See base class.

        Returns:
            List[List[float]]: See base class.
        """
        return self.my_momentum_to_population_transform

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

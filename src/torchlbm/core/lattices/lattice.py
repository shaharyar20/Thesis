import abc
from typing import List


class Lattice(metaclass=abc.ABCMeta):
    """A pure virtual baseclass presribing the methods a velocity set realization has to provide.
    Velocity sets are used to discretize a velocity distribution function.

    Args:
        metaclass (ABCMeta, optional): Baseclass to prescribe abstract methods. Defaults to abc.ABCMeta.
    """

    def __init__(self) -> None:
        """The empty initializer for a velocity set."""
        pass

    @abc.abstractmethod
    def number_of_discrete_velocities(self) -> int:
        """Returns the number of discrete velocity.

        Returns:
            int: The number of discrete velocities.
        """
        pass

    @abc.abstractmethod
    def lattice_velocities(self) -> List[List[float]]:
        """Returns the lattice velocities as a two-dimensional list. Thus, conversion to arbitrary tensor types is possible.

        Returns:
            List[List[float]]: The lattice velocities as a two-dimensional list.
        """
        pass

    @abc.abstractmethod
    def lattice_indices(self) -> List[int]:
        """Returns the indices of the respective velocity directions as a list.

        Returns:
            List[int]: The list of lattice indices.
        """
        pass

    @abc.abstractmethod
    def opposite_lattice_indices(self) -> List[int]:
        """Returns the index of the velocity pointing in the opposite direction as the velocity
        given by the indices of the respective velocity directions. Mainly used to impose boundary conditions.

        Returns:
            List[int]: The opposite lattice indices as a List.
        """
        pass

    @abc.abstractmethod
    def lattice_weights(self) -> List[float]:
        """Return the weights of the respective directions as a List. Can thus be converted to tensors of arbitrary type.

        Returns:
            List[float]: The lattice weights.
        """
        pass

    @abc.abstractmethod
    def population_to_momentum_transform(self) -> List[List[float]]:
        """Returns the matrix that transforms from the population to the momentum space.

        Returns:
            List[List[float]]: The matrix as a List of Lists.
        """
        pass

    @abc.abstractmethod
    def momentum_to_population_transform(self) -> List[List[float]]:
        """Returns the matrix that transforms from the momentum to the population space.

        Returns:
            List[List[float]]: The matrix as a List of Lists.
        """
        pass

    @abc.abstractmethod
    def east_velocities(self) -> List[int]:
        """Returns the lattice indices which contain a velocity component pointing in east direction.

        Returns:
            List[int]: The lattice indices containing a component pointing in east direction.
        """
        pass

    @abc.abstractmethod
    def west_velocities(self) -> List[int]:
        """Returns the lattice indices which contain a velocity component pointing in west direction.

        Returns:
            List[int]: The lattice indices containing a component pointing in west direction.
        """
        pass

    @abc.abstractmethod
    def north_velocities(self) -> List[int]:
        """Returns the lattice indices which contain a velocity component pointing in north direction.

        Returns:
            List[int]: The lattice indices containing a component pointing in north direction.
        """
        pass

    @abc.abstractmethod
    def south_velocities(self) -> List[int]:
        """Returns the lattice indices which contain a velocity component pointing in south direction.

        Returns:
            List[int]: The lattice indices containing a component pointing in south direction.
        """
        pass

    @abc.abstractmethod
    def top_velocities(self) -> List[int]:
        """Returns the lattice indices which contain a velocity component pointing in top direction.

        Returns:
            List[int]: The lattice indices containing a component pointing in top direction.
        """
        pass

    @abc.abstractmethod
    def bottom_velocities(self) -> List[int]:
        """Returns the lattice indices which contain a velocity component pointing in bottom direction.

        Returns:
            List[int]: The lattice indices containing a component pointing in bottom direction.
        """
        pass

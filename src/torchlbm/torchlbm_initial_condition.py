import abc
from typing import List

import torch

from torchlbm.state import TorchlbmSetup


class TorchlbmInitialCondition(metaclass=abc.ABCMeta):
    """Base class to prescribe an initial condition for the macroscopic field and a possibly existing immersed-boundary mesh.

    Args:
        metaclass (ABCmeta, optional): Base class used to prescribe purely virtual functions. Defaults to abc.ABCMeta.
    """

    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        """The initializer for an initial condition.

        Args:
            torchlbm_setup (TorchlbmSetup): The simulation setup.
        """
        self.torchlbm_setup = torchlbm_setup

    @abc.abstractmethod
    def get_initial_velocity(self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor) -> List[torch.Tensor]:
        """Returns the initial velocity. It can be calculated based on the X, Y, and Z coordinate. Has to be implemented by the child class.

        Args:
            X (torch.Tensor): The X coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Y (torch.Tensor): The Y coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Z (torch.Tensor): The Z coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
        """
        pass

    @abc.abstractmethod
    def get_initial_density(self, X, Y, Z) -> torch.Tensor:
        """Returns the initial density. It can be calculated based on the X, Y, and Z coordinate. Has to be implemented by the child class.

        Args:
            X (torch.Tensor): The X coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Y (torch.Tensor): The Y coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Z (torch.Tensor): The Z coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
        """
        pass

    def get_initial_temperature(self, X, Y, Z) -> torch.Tensor:
        """Returns the initial temperature. It can be calculated based on the X, Y, and Z coordinate.

        Args:
            X (torch.Tensor): The X coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Y (torch.Tensor): The Y coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Z (torch.Tensor): The Z coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
        """
        pass

    def get_bounce_back_mask(self, X, Y, Z) -> torch.Tensor:
        """Returns a tensor that contains the mask for which the bounce back boundary condition is applied.
        Contains ones in case the cell is a solid cell and zero otherwises.

        Args:
            X (torch.Tensor): The X coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Y (torch.Tensor): The Y coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Z (torch.Tensor): The Z coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.

        Returns:
            torch.Tensor: The mask containing one if solid cell and zero otherwise. The mask is a (Tx, Ty, Tz) tensor,
                          where Tx, Ty, and Tz denote the total number of the computational domain.
        """
        return None

    def get_curve_function(self, X, Y, Z):
        """Returns a function f(x, y) that defines the boundary as f(x, y, z) = 0.

        Args:
            X (torch.Tensor): The X coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Y (torch.Tensor): The Y coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.
            Z (torch.Tensor): The Z coordinate as a (Tx, Ty, Tz) tensor,
                              where Tx, Ty, and Tz denote the total number of the computational domain.

        Returns:
            function: A function f(x, y) that defines the immersed boundary as f(x, y) = 0.
        """
        return None

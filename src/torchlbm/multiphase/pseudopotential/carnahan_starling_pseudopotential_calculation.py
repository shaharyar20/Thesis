import torch
import torch.nn as nn
import math

from typing import Optional

class CarnahanSterlingPseudopotentialCalculationModule(nn.Module):
    """PyTorch module that calculates the pseudopotential based on the improved Carnahan-Sterling equation of state.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(
        self,
        reduced_temperature: float,
        solid_density: float,
    ) -> None:
        """Constructor of the module. The constructor is usually called from a factory function.

        Args:
            multiphase_active (bool): Indicate whether multiphase is active.
            reduced_temperature (bool): The non-dimensional temperature (normalized wrt to critical temperature).
            solid_density (float): The density of solid nodes prescribed for contact angle.
            bounce_back_mask (torch.Tensor): The mask used for simulating contact angle dynamics. 0 indicates a fluid node,
            1 indicates a solid bounce-back node and 2 indicates a solid wall node.
        """
        super(CarnahanSterlingPseudopotentialCalculationModule, self).__init__()
        self.G = -1.0
        self.reduced_temperature = reduced_temperature
        self.solid_density = solid_density

    def forward(self, density: torch.Tensor, bounce_back_mask: torch.Tensor, temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The main functionality of the module as the forward pass of the module.
        It performs the calculation of the pseudopotential forces.

        Args:
            node_data (NodeData): The node data that contains all storage intense field information.

        Returns:
            torch.Tensor: The calculated pseudopotential.
        """

        density_squared = density**2
        if temperature is None:
            temperature = self.reduced_temperature*0.09433

        eos_nonideal = density*(temperature*(1 + (4*density - 2*density_squared)/(torch.ones_like(density) - density)**3) - density - 1.0/3.0)
        pseudopotential = torch.sqrt(torch.abs(6*eos_nonideal/self.G))

        # if bounce_back_mask is not None:
        #     eos_nonideal_solid = self.solid_density*(temperature*(1 + (4*self.solid_density - 2*self.solid_density*self.solid_density)/(1 - self.solid_density)**3) - self.solid_density - 1.0/3.0)
        #     pseudopotential_solid = math.sqrt(abs(6 * eos_nonideal_solid / self.G))
        #     pseudopotential = torch.where(
        #         bounce_back_mask > 0,
        #         pseudopotential_solid,
        #         pseudopotential,
        #     )

        return pseudopotential

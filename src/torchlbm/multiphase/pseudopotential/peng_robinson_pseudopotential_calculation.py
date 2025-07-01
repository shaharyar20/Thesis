import torch
import torch.nn as nn
import math

from typing import Optional

class PengRobinsonPseudopotentialCalculationModule(nn.Module):
    """PyTorch module that calculates the pseudopotential based on the improved Carnahan-Sterling equation of state.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(
        self,
        reduced_temperature: float,
    ) -> None:
        """Constructor of the module. The constructor is usually called from a factory function.

        Args:
            multiphase_active (bool): Indicate whether multiphase is active.
            reduced_temperature (bool): The non-dimensional temperature (normalized wrt to critical temperature).
            solid_density (float): The density of solid nodes prescribed for contact angle.
            bounce_back_mask (torch.Tensor): The mask used for simulating contact angle dynamics. 0 indicates a fluid node,
            1 indicates a solid bounce-back node and 2 indicates a solid wall node.
        """
        super(PengRobinsonPseudopotentialCalculationModule, self).__init__()
        self.G = -1.0
        self.reduced_temperature = reduced_temperature
        self.critical_temperature = 0.10938
        self.a = 3.0 / 49.0
        self.b = 2.0 / 21.0

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
            temperature = self.reduced_temperature*self.critical_temperature

        acentric_factor = 0.344
        acentric_term = (1 + (0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor**2) * (1 - math.sqrt(temperature/self.critical_temperature)))**2
        eos_nonideal = (density * temperature) / (1 - self.b * density) - (self.a * density_squared * acentric_term) / (1 + 2*self.b*density - self.b**2 * density_squared) - density/3.0

        pseudopotential = torch.sqrt(torch.abs(6*eos_nonideal/self.G))

        # if self.bounce_back_mask_const is not None:
        #     eos_nonideal_solid = self.solid_density*(temperature*(1 + (4*self.solid_density - 2*self.solid_density*self.solid_density)/(1 - self.solid_density)**3) - self.solid_density - 1.0/3.0)
        #     pseudopotential_solid = math.sqrt(abs(6 * eos_nonideal_solid / self.G))
        #     pseudopotential = torch.where(
        #         self.bounce_back_mask_const > 0,
        #         pseudopotential_solid,
        #         pseudopotential,
        #     )

        return pseudopotential

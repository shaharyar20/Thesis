import torch
import torch.nn as nn
import math

from typing import Optional, List

class CarnahanSterlingThermalCalculationModule(nn.Module):
    """PyTorch module that calculates the pseudopotential based on the improved Carnahan-Sterling equation of state.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(
        self,
        Cv: float,
        lattice_weights: List[float],
    ) -> None:
        """Constructor of the module. The constructor is usually called from a factory function.

        Args:
            multiphase_active (bool): Indicate whether multiphase is active.
            reduced_temperature (bool): The non-dimensional temperature (normalized wrt to critical temperature).
            solid_density (float): The density of solid nodes prescribed for contact angle.
            bounce_back_mask (torch.Tensor): The mask used for simulating contact angle dynamics. 0 indicates a fluid node,
            1 indicates a solid bounce-back node and 2 indicates a solid wall node.
        """
        super(CarnahanSterlingThermalCalculationModule, self).__init__()
        self.G = -1.0
        self.Cv = Cv
        self.lattice_weights = torch.tensor(lattice_weights)
        self.register_buffer("lattice_weights_const", self.lattice_weights)

    def forward(self, density: torch.Tensor, temperature: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """The main functionality of the module as the forward pass of the module.
        It performs the calculation of the pseudopotential forces.

        Args:
            node_data (NodeData): The node data that contains all storage intense field information.

        Returns:
            torch.Tensor: The calculated pseudopotential.
        """

        density_squared = density**2
        dp_dT = density*((1 + (4*density - 2*density_squared)/(torch.ones_like(density) - density)**3) - density)

        vx, vy, vz = velocity[0], velocity[1], velocity[2]

        # Use central differences for interior, forward/backward for boundaries
        dvx_dx = torch.zeros_like(vx)
        dvy_dy = torch.zeros_like(vy)
        dvz_dz = torch.zeros_like(vz)

        dvx_dx[1:-1, :, :] = (vx[2:, :, :] - vx[:-2, :, :]) / 2
        dvy_dy[:, 1:-1, :] = (vy[:, 2:, :] - vy[:, :-2, :]) / 2
        dvz_dz[:, :, 1:-1] = (vz[:, :, 2:] - vz[:, :, :-2]) / 2

        # Optionally use forward/backward differences at boundaries (here: zeros)
        # You can modify boundaries as needed (e.g., forward/backward diff)

        divergence = dvx_dx + dvy_dy + dvz_dz
        print(divergence.shape)

        term = temperature*(1 - dp_dT/(density*self.Cv))*divergence
        print(term.shape)

        term = self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * term.unsqueeze(0)
        print(term.shape)
        print(a)

        # eos_nonideal = density*(temperature*(1 + (4*density - 2*density_squared)/(torch.ones_like(density) - density)**3) - density - 1.0/3.0)
        # pseudopotential = torch.sqrt(torch.abs(6*eos_nonideal/self.G))

        # if self.bounce_back_mask_const is not None:
        #     eos_nonideal_solid = self.solid_density*(temperature*(1 + (4*self.solid_density - 2*self.solid_density*self.solid_density)/(1 - self.solid_density)**3) - self.solid_density - 1.0/3.0)
        #     pseudopotential_solid = math.sqrt(abs(6 * eos_nonideal_solid / self.G))
        #     pseudopotential = torch.where(
        #         self.bounce_back_mask_const > 0,
        #         pseudopotential_solid,
        #         pseudopotential,
        #     )

        return pseudopotential

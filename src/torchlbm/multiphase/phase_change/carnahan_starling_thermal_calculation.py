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
        lattice_velocities: List[List[float]],
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
        self.lattice_velocities = torch.tensor(lattice_velocities).clone().detach()
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.kron = torch.einsum('aQ, bQ-> Qab', self.lattice_velocities, self.lattice_velocities)
        self.register_buffer("kron_const", self.kron)

    def forward(self, density: torch.Tensor, temperature: torch.Tensor, velocity: torch.Tensor, old_population: torch.Tensor, new_population: torch.Tensor, vel_omega: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """The main functionality of the module as the forward pass of the module.
        It performs the calculation of the pseudopotential forces.

        Args:
            node_data (NodeData): The node data that contains all storage intense field information.

        Returns:
            torch.Tensor: The calculated pseudopotential.
        """
        macroscopic_velocity = velocity + 0.5 * force / density.unsqueeze(0)

        density_squared = density**2
        dp_dT = density*((1 + (4*density - 2*density_squared)/(torch.ones_like(density) - density)**3) - density)

        # vx, vy, vz = macroscopic_velocity[0], macroscopic_velocity[1], macroscopic_velocity[2]

        # Use central differences for interior, forward/backward for boundaries
        # dvx_dx = torch.zeros_like(vx)
        # dvy_dy = torch.zeros_like(vy)
        # dvz_dz = torch.zeros_like(vz)

        # print(dvx_dx[2:-2, :, :].shape, vx[3:-1, :, :].shape, vx[1:-3, :, :].shape)
        # dvx_dx[2:-2, :, :] = (vx[3:-1, :, :] - vx[1:-3, :, :]) / 2
        # dvy_dy[:, 2:-2, :] = (vy[:, 3:-1, :] - vy[:, 1:-3, :]) / 2
        # dvz_dz[:, :, 2:-2] = (vz[:, :, 3:] - vz[:, :, :-3]) / 2

        # dvx_dx[1:-1, :, :] = (vx[2:, :, :] - vx[:-2, :, :]) / 2
        # dvy_dy[:, 1:-1, :] = (vy[:, 2:, :] - vy[:, :-2, :]) / 2
        # dvz_dz[:, :, 1:-1] = (vz[:, :, 2:] - vz[:, :, :-2]) / 2

        # Optionally use forward/backward differences at boundaries (here: zeros)
        # You can modify boundaries as needed (e.g., forward/backward diff)

        # divergence = dvx_dx + dvy_dy + dvz_dz
        # print(torch.max(divergence))
        # print(torch.min(divergence))   
        # print(torch.max(dvx_dx), torch.min(dvx_dx))
        # print(torch.max(dvy_dy), torch.min(dvy_dy))
        # print(torch.max(dvz_dz), torch.min(dvz_dz))

        # Divergence using strain rate tensor
        neq = old_population - new_population

        second_moment = torch.einsum(
            "QNML,Qde->deNML",
            neq,
            self.kron_const,
        )
        strain_rate_tensor = - 3 * vel_omega * second_moment / (2 * density)
        forcing_tensor = torch.einsum("ixyz, jxyz->ijxyz", force, macroscopic_velocity) + torch.einsum("jxyz, ixyz->ijxyz", force, macroscopic_velocity)
        forcing_tensor = -forcing_tensor * 3 * vel_omega / (4 * density)
        strain_rate_tensor = strain_rate_tensor + forcing_tensor
        div = strain_rate_tensor[0, 0, :, :, :] + strain_rate_tensor[1, 1, :, :, :] + strain_rate_tensor[2, 2, :, :, :]
        # print(div.shape)
        # print(torch.max(force))
        # print(torch.max(div))
        # print(torch.min(div))
        # print(torch.max(strain_rate_tensor[0, 0, :, :, :]), torch.min(strain_rate_tensor[0, 0, :, :, :]))
        # print(torch.max(strain_rate_tensor[1, 1, :, :, :]), torch.min(strain_rate_tensor[1, 1, :, :, :]))
        # print(torch.max(strain_rate_tensor[2, 2, :, :, :]), torch.min(strain_rate_tensor[2, 2, :, :, :]))
        # print(a)

        term = temperature*(1 - dp_dT/(density*self.Cv))*div
        # print(term.shape)
        

        term = self.lattice_weights_const.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * term.unsqueeze(0)
        # print(term.shape)
        # print(torch.max(term), torch.min(term))
        # print(a)

        return term

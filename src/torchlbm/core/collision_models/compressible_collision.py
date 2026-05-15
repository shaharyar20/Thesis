import torch
import torch.nn as nn
from dataclasses import dataclass

# from torchlbm.node_data import NodeData
# from torchlbm.thermal_node_data import ThermalNodeData
from torchlbm.compressible_node_data import CompressibleNodeData

def calc_temp_weights(temp):
    temp_0 = (1 - temp)**2
    temp_1234 = (1 - temp) * temp / 2.0
    temp_5678 = (temp / 2.0)**2
    weights = torch.stack([temp_0, temp_1234, temp_1234, temp_1234, temp_1234, temp_5678, temp_5678, temp_5678, temp_5678], dim=0)
    return weights

class CompressibleCollisionModule(nn.Module):
    """A Torch LBM module that performs the single relaxation time collision step of a Lattice-Boltzmann algorithm

    Args:
        nn (nn.Module): The collision operator is implemented as a PyTorch module. This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(self, viscosity: float, cp: float, thermal_conductivity: float, lattice_velocities, shifted_velx, shifted_vely) -> None:
        """The initializer of the collision module.

        Args:
            relaxation_omega (float): The relaxation frequency.
        """
        super(CompressibleCollisionModule, self).__init__()
        self.viscosity = viscosity
        self.cp = cp
        self.thermal_conductivity = thermal_conductivity
        self.lattice_velocities = torch.tensor(lattice_velocities)
        self.shifted_velx = shifted_velx
        self.shifted_vely = shifted_vely
        self.lattice_velocities[0] += self.shifted_velx
        self.lattice_velocities[1] += self.shifted_vely
        print(self.lattice_velocities)
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        """The forward pass of the collision modules. Gets as input the discretized velocity distribution of the start of the timestept,
        and the equilibrium distribution calculated based on it. It returns the post-collision distribution.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            torch.Tensor: The discretized velocity distribution after collision.
        """
        # print(node_data.distributions.vel_old_population[:, 1500, 2], node_data.distributions.vel_old_population[:, 1501, 2])

        # node_data.moments.temperature = torch.ones_like(node_data.moments.temperature) * 0.1
        # print("Before collision: ", node_data.distributions.temp_old_population.shape)
        #self.relaxation_omega_vel = 1.0 / ((self.viscosity / (node_data.moments.density * node_data.moments.temperature)) + 0.5)
        #self.relaxation_omega_temp = 1.0 / ((self.thermal_conductivity / (self.cp * node_data.moments.density * node_data.moments.temperature)) + 0.5)
       
       
       
        # 1. Calculate base relaxation frequencies and tau
        self.relaxation_omega_vel = 1.0 / ((self.viscosity / (node_data.moments.density * node_data.moments.temperature)) + 0.5)
        self.relaxation_omega_temp = 1.0 / ((self.thermal_conductivity / (self.cp * node_data.moments.density * node_data.moments.temperature)) + 0.5)
        
        tau_vel = 1.0 / self.relaxation_omega_vel
        tau_temp = 1.0 / self.relaxation_omega_temp

        # 2. Calculate the local Knudsen sensor (epsilon)
        # f_i is vel_old_population, f_eq is vel_new_population (dim=0 is the Q discrete directions)
        # Adding 1e-8 to the denominator prevents Division By Zero in empty/solid nodes
        f_curr = node_data.distributions.vel_old_population
        f_eq = node_data.distributions.vel_new_population
        
        epsilon = torch.mean(torch.abs(f_curr - f_eq) / (torch.abs(f_eq) + 1e-8), dim=0, keepdim=True)

        # 3. Create the piecewise alpha multiplier
        alpha = torch.ones_like(epsilon)
        
        # We use torch.where to apply the piecewise conditions completely in parallel
        alpha = torch.where((epsilon >= 0.01) & (epsilon < 0.10), torch.tensor(1.05, device=epsilon.device, dtype=epsilon.dtype), alpha)
        alpha = torch.where((epsilon >= 0.10) & (epsilon < 1.0),  torch.tensor(1.35, device=epsilon.device, dtype=epsilon.dtype), alpha)
        
        # The Nuclear Option: if epsilon >= 1.0, alpha = 1 / tau, forcing tau_new = 1.0 (omega_new = 1.0)
        alpha = torch.where(epsilon >= 1.0, 1.0 / tau_vel, alpha)

        # 4. Apply the multiplier to both fluid and thermal relaxation times
        self.relaxation_omega_vel = 1.0 / (alpha * tau_vel)
        self.relaxation_omega_temp = 1.0 / (alpha * tau_temp)





        node_data.distributions.vel_old_population = torch.where(
            node_data.bounce_back_mask > 0,
            node_data.distributions.vel_old_population,
            (1.0 - self.relaxation_omega_vel) * node_data.distributions.vel_old_population + self.relaxation_omega_vel * node_data.distributions.vel_new_population
        )

        pressure_tensor = torch.einsum("QNML, dQ, eQ -> deNML", node_data.distributions.vel_old_population, self.lattice_velocities_const, self.lattice_velocities_const)
        pressure_tensor_eq = torch.einsum("QNML, dQ, eQ -> deNML", node_data.distributions.vel_new_population, self.lattice_velocities_const, self.lattice_velocities_const)
        # print((pressure_tensor - pressure_tensor_eq).shape)
        quasi_eq = torch.einsum("aNML, aQ -> QNML", torch.einsum("bNML,abNML->aNML", node_data.moments.velocity, pressure_tensor-pressure_tensor_eq), self.lattice_velocities_const)
        temp_weights = calc_temp_weights(node_data.moments.temperature)
        quasi_eq *= (2 * temp_weights / node_data.moments.temperature)

        node_data.distributions.temp_old_population = torch.where(
            node_data.bounce_back_mask > 0,
            node_data.distributions.temp_old_population,
            (1.0 - self.relaxation_omega_vel) * node_data.distributions.temp_old_population + self.relaxation_omega_vel * node_data.distributions.temp_new_population + (self.relaxation_omega_vel - self.relaxation_omega_temp) * quasi_eq
        )

        # print(node_data.distributions.vel_old_population[:, 1500, 2], node_data.distributions.vel_old_population[:, 1501, 2])

        node_data.distributions.temp_old_population[:, 1:, :, :] = self.shifted_velx * node_data.distributions.temp_old_population[:, :-1, :, :] + (1 - self.shifted_velx) * node_data.distributions.temp_old_population[:, 1:, :, :]
        node_data.distributions.vel_old_population[:, 1:, :, :] = self.shifted_velx * node_data.distributions.vel_old_population[:, :-1, :, :] + (1 - self.shifted_velx) * node_data.distributions.vel_old_population[:, 1:, :, :]
        # node_data.distributions.temp_old_population[:, :, 1:, :] = self.shifted_vely * node_data.distributions.temp_old_population[:, :, :-1, :] + (1 - self.shifted_vely) * node_data.distributions.temp_old_population[:, :, 1:, :]
        # node_data.distributions.vel_old_population[:, :, 1:, :] = self.shifted_vely * node_data.distributions.vel_old_population[:, :, :-1, :] + (1 - self.shifted_vely) * node_data.distributions.vel_old_population[:, :, 1:, :]

        #Bilinear interpolation
        # node_data.distributions.temp_old_population[:, 1:, 1:, :] = (
        #     (1 - self.shifted_velx) * (1 - self.shifted_vely) * node_data.distributions.temp_old_population[:, 1:, 1:, :] +
        #     (1 - self.shifted_velx) * self.shifted_vely * node_data.distributions.temp_old_population[:, 1:, :-1, :] +
        #     self.shifted_velx * (1 - self.shifted_vely) * node_data.distributions.temp_old_population[:, :-1, 1:, :] +
        #     self.shifted_velx * self.shifted_vely * node_data.distributions.temp_old_population[:, :-1, :-1, :]
        # )
        # node_data.distributions.vel_old_population[:, 1:, 1:, :] = (
        #     (1 - self.shifted_velx) * (1 - self.shifted_vely) * node_data.distributions.vel_old_population[:, 1:, 1:, :] +
        #     (1 - self.shifted_velx) * self.shifted_vely * node_data.distributions.vel_old_population[:, 1:, :-1, :] +
        #     self.shifted_velx * (1 - self.shifted_vely) * node_data.distributions.vel_old_population[:, :-1, 1:, :] +
        #     self.shifted_velx * self.shifted_vely * node_data.distributions.vel_old_population[:, :-1, :-1, :]
        # )

        # print(node_data.distributions.vel_old_population[:, 1500, 2], node_data.distributions.vel_old_population[:, 1501, 2])

        # print("After collision: ", node_data.distributions.temp_old_population.shape)

        # print(self.relaxation_omega_temp[5,5,5,5])

        # discrete_velocities_post_collision = (
        #     1.0 - self.relaxation_omega_vel
        # ) * node_data.distributions.vel_old_population + self.relaxation_omega_vel * node_data.distributions.vel_new_population
        # discrete_temperatures_post_collision = (
        #     1.0 - self.relaxation_omega_temp
        # ) * node_data.distributions.temp_old_population + self.relaxation_omega_temp * node_data.distributions.temp_new_population
        return node_data

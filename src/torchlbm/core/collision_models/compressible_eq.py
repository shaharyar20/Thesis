from typing import List

import torch.nn as nn
import torch
from dataclasses import dataclass
from functools import partial
from torch.func import jacrev, vmap, jacfwd

# from torchlbm.node_data import NodeData
from torchlbm.compressible_node_data import CompressibleNodeData

def calc_vel_weights(velx, vely, temp):
    velx_sq = velx * velx
    vely_sq = vely * vely
    velx_pos = 0.5 * (velx + velx_sq + temp)
    velx_neg = 0.5 * (-velx + velx_sq + temp)
    velx_zero = 1 - velx_sq - temp
    vely_pos = 0.5 * (vely + vely_sq + temp)
    vely_neg = 0.5 * (-vely + vely_sq + temp)
    vely_zero = 1 - vely_sq - temp
    weights = torch.stack([velx_zero*vely_zero, velx_pos*vely_zero, velx_zero*vely_pos, velx_neg*vely_zero, velx_zero*vely_neg, velx_pos*vely_pos, velx_neg*vely_pos, velx_neg*vely_neg, velx_pos*vely_neg], dim=0)
    return weights

def calc_temp_weights(temp):
    temp_0 = (1 - temp)**2
    temp_1234 = (1 - temp) * temp / 2.0
    temp_5678 = (temp / 2.0)**2
    weights = torch.stack([temp_0, temp_1234, temp_1234, temp_1234, temp_1234, temp_5678, temp_5678, temp_5678, temp_5678], dim=0)
    return weights

def batched_equations(a, b, c, weights, rho, ux, uy, T, E, v):
    exp_term = rho * weights * torch.exp(a+b*v[0]+c*v[1])
    F1 = torch.sum(exp_term) - 2 * rho * E
    F2 = torch.sum(v[0]*exp_term) - 2 * rho * ux * (E + T)
    F3 = torch.sum(v[1]*exp_term) - 2 * rho * uy * (E + T)
    return torch.stack([F1, F2, F3])

def full_equations(a, b, c, weights, rho, ux, uy, T, E, v):
    exp_term = rho * weights * torch.exp(a + b*v[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + c*v[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
    F1 = torch.sum(exp_term, dim=0) - 2 * rho * E
    F2 = torch.sum(v[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*exp_term, dim=0) - 2 * rho * ux * (E + T)
    F3 = torch.sum(v[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)*exp_term, dim=0) - 2 * rho * uy * (E + T)
    return torch.stack([F1, F2, F3])

class CompressibleEquilibriumCalculationModule(nn.Module):
    """PyTorch module that implements the calculation of the equilibrium distribution based on a linear BGK operator.
    It calculates the equilibrium for the Navier-Stokes equations.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used ase base class.
    """

    def __init__(self, lattice_velocities: List[List[float]], lattice_weights: List[float], shifted_velx, shifted_vely) -> None:
        """Initializer for the equilibrium calculation module.

        Args:
            lattice_velocities (List[List[float]]): The lattice velocities of the used velocity set.
            lattice_weights (List[float]): The lattice weights of the used velocity set.
        """
        super(CompressibleEquilibriumCalculationModule, self).__init__()
        self.lattice_velocities = torch.tensor(lattice_velocities)
        self.shifted_velx = shifted_velx
        self.shifted_vely = shifted_vely
        self.lattice_velocities[0] += self.shifted_velx
        self.lattice_velocities[1] += self.shifted_vely
        print(self.lattice_velocities)
        self.register_buffer("lattice_velocities_const", self.lattice_velocities)
        self.lattice_weights = torch.tensor(lattice_weights)
        self.register_buffer("lattice_weights_const", self.lattice_weights)
        self.max_iters = 10
        self.tol = 1e-6
    

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        """The forward pass of the equilibrium calculation module. The equilibrium distribution for the Navier-Stokes equations are calculated.
        It gets as input macroscopic quantities, the moments of the velocity distribution (density and velocity) and calculates the
        equilibrium distribution based on them.

        Args:
            node_data (NodeData): The node_data object is a pure data container that contains the storage heavy macroscopic and microscopix field data.

        Returns:
            NodeData: The modified node_data object where the new_population is overwritten with the equilibrium distribution.
        """

        
        rho = node_data.moments.density # X, Y, 1
        E = node_data.moments.energy # X, Y, 1
        ux = node_data.moments.velocity[0] # X, Y, 1
        uy = node_data.moments.velocity[1] # X, Y, 1
        temp = node_data.moments.temperature # X, Y, 1
        # temp = 0.1 * torch.ones_like(node_data.moments.temperature)
        # print(rho.shape, E.shape, ux.shape, uy.shape, temp.shape)

        # E = 0.5 * (ux**2 + uy**2) + temp # X, Y
        shifted_ux = ux - self.shifted_velx
        shifted_uy = uy - self.shifted_vely

        vel_weights = calc_vel_weights(shifted_ux, shifted_uy, temp) # 9, X, Y, 1
        node_data.distributions.vel_new_population = rho * vel_weights # 9, X, Y, 1
        # print(vel_weights.shape, f_eq.shape)

        Q, X, Y = vel_weights.squeeze(-1).shape
        # print(Q, X, Y)

        # Lagrange multipliers
        a = torch.zeros((X, Y, 1), dtype=torch.float16, requires_grad=True, device=rho.device)
        b = torch.zeros((X, Y, 1), dtype=torch.float16, requires_grad=True, device=rho.device)
        c = torch.zeros((X, Y, 1), dtype=torch.float16, requires_grad=True, device=rho.device)
        
        # Lattice velocities
        v = self.lattice_velocities_const[:2]
        # v[0] = v[0] + self.shifted_velx
        # v[1] = v[1] + self.shifted_vely
        batched_equations_with_v = partial(batched_equations, v=v)
        
        temp_weights = calc_temp_weights(temp)

        # print(f"Memory before eq: {torch.cuda.memory_allocated()/(1024 ** 2)}")

        for i in range(self.max_iters):
            # memory1 = torch.cuda.memory_allocated()/(1024 ** 2)
            # with torch.no_grad():
            jac_tuples = vmap(vmap(jacfwd(batched_equations_with_v, argnums=(0,1,2))))(a, b, c, temp_weights.squeeze(-1).permute(1,2,0), rho, ux, uy, temp, E)
            jac = torch.stack([jac_tuples[0], jac_tuples[1], jac_tuples[2]], dim=3).squeeze(-1).squeeze(-1)
            full = full_equations(a, b, c, temp_weights, rho, ux, uy, temp, E, v).squeeze(-1).permute(1,2,0)
            # memory2 = torch.cuda.memory_allocated()/(1024 ** 2)

            with torch.no_grad():
                delta = torch.linalg.solve(jac, -full.detach())
                a = a + delta[:,:,0].unsqueeze(-1)
                b = b + delta[:,:,1].unsqueeze(-1)
                c = c + delta[:,:,2].unsqueeze(-1)

            # print(torch.max(torch.norm(delta, dim=2)))
            # memory3 = torch.cuda.memory_allocated()/(1024 ** 2)
            # print(i, memory2 - memory1, memory3 - memory2)

            if torch.max(torch.norm(delta, dim=2)) < self.tol:
                # print(torch.max(torch.norm(delta, dim=2)))
                break

        # print(f"Memory after eq: {torch.cuda.memory_allocated()/(1024 ** 2)}")

        node_data.distributions.temp_new_population = rho * temp_weights * torch.exp(a + b * v[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + c * v[1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) # 9, X, Y, 1

        # print(vel_weights[10])
        return node_data

import torch
import math

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup


class TaylorGreenVortexInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        u_0 = 0.3e-2
        U_physical = (
            u_0
            * torch.cos(X)
            * torch.sin(Y)
            # * torch.sin(2.0 * torch.pi * Z)
        )
        V_physical = (
            -u_0
            * torch.sin(X)
            * torch.cos(Y)
            # * torch.sin(2.0 * torch.pi * Z)
        )
        W_physical = torch.zeros_like(X)
        return [
            U_physical,
            V_physical,
            W_physical,
        ]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X)
    
    def get_bounce_back_mask(self, X, Y, Z):
        mask = torch.zeros_like(X).bool()
        
    #     n_circles = 50
    #     torch.random.seed = 200
    #     for i in range(n_circles):
    #         x0 = torch.rand([1]).item()
    #         y0 = torch.rand([1]).item()
    #         r = 0.02 + 0.035 * torch.rand([1]).item()
    #         mask = torch.where(torch.sqrt((X-x0)*(X-x0)+(Y-y0)*(Y-y0)) < r, 1, mask)


        return mask


def main():

    simulation_setup = TorchlbmSetup("TaylorGreenVortex")
    simulation_setup["Domain"]["Dimension"].value = "2D"
    simulation_setup["Domain"]["NodeSize"].value = 2 * torch.pi
    simulation_setup["Domain"]["CellsPerNode"].value = 200
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    simulation_setup["Output"]["Active"].value = True
    simulation_setup["Output"]["ProfilingActive"].value = False
    simulation_setup["Output"]["OutputTimeInterval"].value = 20.0
    simulation_setup["Output"]["OutputEveryStep"].value = False
    simulation_setup["Output"]["Velocity"]["Active"].value = True
    simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 0.002]
    simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
    simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["Active"].value = True
    simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]

    simulation_setup["Physics"]["MachNumber"].value = 0.01 * math.sqrt(3.0)
    simulation_setup["Physics"]["EndTime"].value = 60.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 0.01
    simulation_setup["Physics"]["KinematicViscosityPu"].value = 0.05
    simulation_setup["Physics"]["Precision"].value = "Single"

    simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "SRT"

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = TaylorGreenVortexInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False)
    simulation.run()


if __name__ == "__main__":

    main()

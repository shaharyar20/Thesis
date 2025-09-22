import torch
import math

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup


class BumpChannelInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        # U_physical = torch.zeros_like(X)
        U_physical = 0.01 * torch.ones_like(X)
        V_physical = torch.zeros_like(X)
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

        # mask = torch.where((0.5 + 0.15*((1-torch.cos(2*torch.pi*X/2.0))) - Y) < 0, 1, mask)
        mask = torch.where((0.6 +  0.3*torch.sin(torch.pi*X/4.0) - Y) < 0, 1, mask)
        mask = torch.where((0.1 + 0.3*torch.sin(torch.pi*X/4.0) - Y) > 0, 1, mask)
        # mask = torch.where(torch.sqrt((X-0.5)*(X-0.5)+(Y-0.5)*(Y-0.5)) < 0.1, 1, mask)
        return mask
    
    def get_curve_function(self, X, Y, Z):
        # return 0.5 + 0.3*torch.sin(torch.pi*X/4.0) - Y
        top = lambda x, y, z: 0.6 + 0.3*torch.sin(torch.pi*x/4.0) - y
        bottom = lambda x, y, z: 0.1 + 0.3*torch.sin(torch.pi*x/4.0) - y

        return [top, bottom]


def main():

    simulation_setup = TorchlbmSetup("BumpChannel")
    simulation_setup["Domain"]["Dimension"].value = "2D"
    simulation_setup["Domain"]["NodeSize"].value = 1.0
    simulation_setup["Domain"]["CellsPerNode"].value = 80
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [2, 1, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Outlet"
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["OutletDensity"].value = 1.0
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [1.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["BounceBackType"].value = "Halfway"

    # simulation_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value = False
    # simulation_setup["InitialCondition"]["Density"].value = "1.0"
    # simulation_setup["InitialCondition"]["Velocity"]["x"].value = "0.02"
    # simulation_setup["InitialCondition"]["Velocity"]["y"].value = "0.0"
    # simulation_setup["InitialCondition"]["Velocity"]["z"].value = "0.0"
    # simulation_setup["InitialCondition"]["BounceBackMask"].value = f"lambda x, y, z: torch.where(torch.sqrt((x-{1*0.15})*(x-{1*0.15})+(y-{1*0.5})*(y-{1*0.5})) < {0.1}, 1, 0)"
    # simulation_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value = False
    # simulation_setup["InitialCondition"]["PyTorchFields"]["Density"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/density_0.01196723.pt"
    # simulation_setup["InitialCondition"]["PyTorchFields"]["Velocity"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/velocity_0.01196723.pt"
    

    simulation_setup["Output"]["Active"].value = True
    # simulation_setup["Output"]["ModulusArtifactsActive"].value = True
    # simulation_setup["Output"]["PrintTimingInformation"].value = False
    simulation_setup["Output"]["OutputTimeInterval"].value = 10.0
    simulation_setup["Output"]["Velocity"]["Active"].value = True
    simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 1.5]
    simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
    simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["Active"].value = True
    simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
    simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

    simulation_setup["Physics"]["MachNumber"].value = 0.1
    simulation_setup["Physics"]["EndTime"].value = 20.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 1.0
    simulation_setup["Physics"]["KinematicViscosityPu"].value = 0.01 # Make tau 1
    simulation_setup["Physics"]["Precision"].value = "Single"
    # simulation_setup["Physics"]["VolumeForces"]["Active"].value = True
    # simulation_setup["Physics"]["VolumeForces"]["Type"].value = "ShanChen"
    # simulation_setup["Physics"]["VolumeForces"]["ForceVector"].value = [1.0, 0.0, 0.0]

    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "SRT"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = BumpChannelInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False)
    simulation.run()


if __name__ == "__main__":

    main()

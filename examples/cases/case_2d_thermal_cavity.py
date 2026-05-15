import torch
import math

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup


class LidDrivenCavityInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        return [
            torch.zeros_like(X),
            torch.zeros_like(X),
            torch.zeros_like(X),
        ]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X)
    
    def get_initial_temperature(self, X, Y, Z):
        return torch.zeros_like(X)
    
    def get_bounce_back_mask(self, X, Y, Z):
        mask = torch.zeros_like(X).bool()
        return mask


def main():

    simulation_setup = TorchlbmSetup("ThermalLidDrivenCavity")
    simulation_setup["Domain"]["Dimension"].value = "2D"
    simulation_setup["Domain"]["NodeSize"].value = 64.0  #2 * torch.pi
    simulation_setup["Domain"]["CellsPerNode"].value = 64
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["WallVelocity"].value = [0.026, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    simulation_setup["Thermal"]["Active"].value = True
    simulation_setup["Thermal"]["HeatConductivity"].value = 0.5/3.0
    simulation_setup["Thermal"]["BoundaryConditions"]["East"]["WallTemperature"].value = 0.0
    simulation_setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value = 0.0
    simulation_setup["Thermal"]["BoundaryConditions"]["North"]["WallTemperature"].value = 1.0
    simulation_setup["Thermal"]["BoundaryConditions"]["South"]["WallTemperature"].value = 0.0

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
    simulation_setup["Output"]["OutputTimeInterval"].value = 100.0
    simulation_setup["Output"]["OutputEveryStep"].value = False
    # simulation_setup["Output"]["Velocity"]["Active"].value = True
    # simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 0.04]
    # simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = False
    # simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    # simulation_setup["Output"]["Density"]["Active"].value = True
    # simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    # simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
    # simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

    simulation_setup["Physics"]["MachNumber"].value = 0.026 * math.sqrt(3.0)
    simulation_setup["Physics"]["EndTime"].value = 2000.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 0.026
    simulation_setup["Physics"]["KinematicViscosityPu"].value = 0.5/3 #0.00818125 #0.00818125 #0.032725 #0.0163625 # Make tau 1
    simulation_setup["Physics"]["Precision"].value = "Single"
    # simulation_setup["Physics"]["VolumeForces"]["Active"].value = True
    # simulation_setup["Physics"]["VolumeForces"]["Type"].value = "ShanChen"
    # simulation_setup["Physics"]["VolumeForces"]["ForceVector"].value = [1.0, 0.0, 0.0]

    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "SRT"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "distribution_learning/models/consres_egnn_u03_n2k_pos_s5e4.pth"

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = LidDrivenCavityInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False)
    simulation.run()


if __name__ == "__main__":

    main()

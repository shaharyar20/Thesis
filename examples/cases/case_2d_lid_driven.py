import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.io_tools.output_writer import OutputWriter
import torchlbm.standalone_operations.file_operations as file_o

u = 0.05
L = 129
Re = 4000
nu = u * L / Re

class LidDrivenCavityInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        U_physical = torch.zeros_like(X)
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

        return mask
    
class LidDrivenCavityOutputWriter(OutputWriter):
    def __init__(self, result_folder: str, logger, state) -> None:
        super().__init__(result_folder, logger, state)
        self.horizontal_velocity_centerline = None
        self.vertical_velocity_centerline = None

    def evaluate_quantities(self, state, timestamp):
        num_halos = state.torchlbm_setup["Domain"]["NumHaloCells"].value
        u_x = state.node_data.moments.velocity[0, (L-1)//2 + num_halos, num_halos:-num_halos, 0] / u
        u_y = state.node_data.moments.velocity[1, num_halos:-num_halos, (L-1)//2 + num_halos, 0] / u
        self.horizontal_velocity_centerline = u_x.cpu().numpy()
        self.vertical_velocity_centerline = u_y.cpu().numpy()

    def write_quantities(self, state):
        quantity_folder = self._result_folder.joinpath("quantities")
        file_o.create_folder(quantity_folder)
        np.save(quantity_folder.joinpath("horizontal_velocity_centerline.npy"), self.horizontal_velocity_centerline)
        np.save(quantity_folder.joinpath("vertical_velocity_centerline.npy"), self.vertical_velocity_centerline)

        # Plot the horizontal velocity centerline
        plt.figure()
        plt.plot(self.horizontal_velocity_centerline, label="Simulation")
        plt.xlabel("y")
        plt.ylabel("X-velocity")
        plt.title("Horizontal Velocity Centerline")
        plt.legend()
        plt.grid()
        plt.savefig(quantity_folder.joinpath("horizontal_velocity_centerline.png"))
        plt.close()

        # Plot the vertical velocity centerline
        plt.figure()
        plt.plot(self.vertical_velocity_centerline, label="Simulation")
        plt.xlabel("x")
        plt.ylabel("Y-velocity")
        plt.title("Vertical Velocity Centerline")
        plt.legend()
        plt.grid()
        plt.savefig(quantity_folder.joinpath("vertical_velocity_centerline.png"))
        plt.close()


def main():

    simulation_setup = TorchlbmSetup("LidDrivenCavityOutput")
    simulation_setup["Domain"]["Dimension"].value = "2D"
    simulation_setup["Domain"]["NodeSize"].value = L
    simulation_setup["Domain"]["CellsPerNode"].value = L
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [1, 1, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["WallVelocity"].value = [u, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

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
    simulation_setup["Output"]["OutputTimeInterval"].value = 6000.0
    simulation_setup["Output"]["Velocity"]["Active"].value = True
    simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 1.0]
    simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = False
    simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["Active"].value = True
    simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    # simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
    # simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["EvaluationTimeInterval"].value = 120000.0

    simulation_setup["Physics"]["MachNumber"].value = u * math.sqrt(3.0)
    simulation_setup["Physics"]["EndTime"].value = 120000.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = u
    simulation_setup["Physics"]["KinematicViscosityPu"].value = nu
    simulation_setup["Physics"]["Precision"].value = "Single"
    # simulation_setup["Physics"]["VolumeForces"]["Active"].value = True
    # simulation_setup["Physics"]["VolumeForces"]["Type"].value = "ShanChen"
    # simulation_setup["Physics"]["VolumeForces"]["ForceVector"].value = [1.0, 0.0, 0.0]

    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "EntropicMRT"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/358668294755187420/cd2ec4e3f9d34dfaa1253842fbf530ac/checkpoints/epoch=126-step=793750.ckpt"

    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = LidDrivenCavityInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False, output_writer=LidDrivenCavityOutputWriter)
    simulation.run()


if __name__ == "__main__":

    main()

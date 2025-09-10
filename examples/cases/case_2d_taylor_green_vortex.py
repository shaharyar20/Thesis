import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.io_tools.output_writer import OutputWriter
import torchlbm.standalone_operations.file_operations as file_o


# Lattice unit parameters
u = 0.05
L = 100
Re = 10
nu = u * L / Re
# omega = 1.95
nu = 0.4 / 3.0

class TaylorGreenVortexInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        U_physical = (
            u
            * torch.cos(2 * torch.pi * X / L)
            * torch.sin(2 * torch.pi * Y / L)
        )
        V_physical = (
            -u
            * torch.sin(2 * torch.pi * X / L)
            * torch.cos(2 * torch.pi * Y / L)
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
        return mask
    
class TaylorGreenVortexOutputWriter(OutputWriter):
    def __init__(self, result_folder: str, logger, state) -> None:
        super().__init__(result_folder, logger, state)
        # Kinetic energy and timestamp lists
        self.kinetic_energy_list = []
        self.timestamp_list = []
        self.initial_kinetic_energy = None
        self.analytical_kinetic_energy_list = []

    def evaluate_quantities(self, state, timestamp):
        # print("Evaluating quantities at time:", timestamp)
        num_halos = state.torchlbm_setup["Domain"]["NumHaloCells"].value
        vel_mag_squared = torch.einsum("DNML, DNML -> NML", 
            state.node_data.moments.velocity[:, num_halos:-num_halos, num_halos:-num_halos, :], 
            state.node_data.moments.velocity[:, num_halos:-num_halos, num_halos:-num_halos, :]
        )
        kinetic_energy = 0.5 * torch.mean(vel_mag_squared).item()
        self.kinetic_energy_list.append(kinetic_energy)
        self.timestamp_list.append(timestamp)
        if self.initial_kinetic_energy is None:
            self.initial_kinetic_energy = kinetic_energy
        # print(self.initial_kinetic_energy)
        k = 2 * math.pi / L
        analytical_kinetic_energy = self.initial_kinetic_energy * math.exp(
            -4*nu * (k**2) * timestamp)
        # print(nu, ((2 * math.pi / L) ** 2), timestamp, analytical_kinetic_energy, kinetic_energy)
        self.analytical_kinetic_energy_list.append(analytical_kinetic_energy)
    
    def write_quantities(self, state):
        quantity_folder = self._result_folder.joinpath("quantities")
        file_o.create_folder(quantity_folder)
        # Create numpy arrays for all lists
        kinetic_energy_array = np.array(self.kinetic_energy_list)
        timestamp_array = np.array(self.timestamp_list)
        analytical_kinetic_energy_array = np.array(self.analytical_kinetic_energy_list)
        # Save arrays to files
        np.save(quantity_folder.joinpath("kinetic_energy.npy"), kinetic_energy_array)
        np.save(quantity_folder.joinpath("timestamp.npy"), timestamp_array)
        np.save(quantity_folder.joinpath("analytical_kinetic_energy.npy"), analytical_kinetic_energy_array)
        # Plot kinetic energy over time
        plt.figure()
        plt.plot(self.timestamp_list, self.kinetic_energy_list, label="Simulated")
        plt.plot(self.timestamp_list, self.analytical_kinetic_energy_list, label="Analytical", linestyle="dashed")
        plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("Kinetic Energy")
        plt.legend()
        plt.grid()
        plt.title("Kinetic Energy Decay")
        plt.savefig(quantity_folder.joinpath("kinetic_energy.png"))
        plt.close()


def main():

    simulation_setup = TorchlbmSetup("TaylorGreenVortexOutput")
    simulation_setup["Domain"]["Dimension"].value = "2D"
    simulation_setup["Domain"]["NodeSize"].value = L
    simulation_setup["Domain"]["CellsPerNode"].value = L
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
    simulation_setup["Output"]["OutputTimeInterval"].value = 3200.0
    simulation_setup["Output"]["OutputEveryStep"].value = False
    simulation_setup["Output"]["Velocity"]["Active"].value = True
    simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 0.05]
    simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
    simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["Active"].value = True
    simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["EvaluationTimeInterval"].value = 50.0
    
    simulation_setup["Physics"]["MachNumber"].value = u * math.sqrt(3.0)
    simulation_setup["Physics"]["EndTime"].value = 3200.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = u
    simulation_setup["Physics"]["KinematicViscosityPu"].value = nu
    simulation_setup["Physics"]["Precision"].value = "Single"

    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "GNN"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/739087019080992631/59f8b1d89fec49a8be21080a1c19d6db/checkpoints/epoch=99-step=625000.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/671044091917114828/e2af73c363b94dd28fbfb06deefab240/checkpoints/epoch=139-step=875000.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/671044091917114828/13790144cb2c4a9d9d37b7f81ae7d995/checkpoints/epoch=132-step=831250.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/358668294755187420/b49f942c27054d248040ef73c1fb7235/checkpoints/epoch=133-step=837500.ckpt"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/358668294755187420/cd2ec4e3f9d34dfaa1253842fbf530ac/checkpoints/epoch=126-step=793750.ckpt"
    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = TaylorGreenVortexInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False, output_writer=TaylorGreenVortexOutputWriter)
    simulation.run()


if __name__ == "__main__":

    main()

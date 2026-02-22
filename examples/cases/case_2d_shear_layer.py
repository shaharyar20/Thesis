import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.io_tools.output_writer import OutputWriter
import torchlbm.standalone_operations.file_operations as file_o

u = 0.04
L = 256
Re = 30000
nu = u * L / Re
k = 80.0
delta = 0.05
end_time = 4 * L / u


class ShearLayerInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        U_physical = torch.where(
            Y >= L / 2.0,
            u * torch.tanh(k * (3.0 / 4.0 - Y / L)),
            u * torch.tanh(k * (Y / L - 1.0 / 4.0)),
        )
        V_physical = u * delta * torch.sin(2.0 * torch.pi * (X / L + 1.0 / 4.0))
        W_physical = torch.zeros_like(V_physical)
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
    
class ShearLayerOutputWriter(OutputWriter):
    def __init__(self, result_folder: str, logger, state) -> None:
        super().__init__(result_folder, logger, state)
        # Kinetic energy and timestamp lists
        self.kinetic_energy_list = []
        self.enstrophy_list = []
        self.entropy_list = []
        self.timestamp_list = []

    def evaluate_quantities(self, state, timestamp):
        num_halos = state.torchlbm_setup["Domain"]["NumHaloCells"].value
        vel_mag_squared = torch.einsum("DNML, DNML -> NML", 
            state.node_data.moments.velocity[:, num_halos:-num_halos, num_halos:-num_halos, :], 
            state.node_data.moments.velocity[:, num_halos:-num_halos, num_halos:-num_halos, :]
        )
        kinetic_energy = 0.5 * torch.mean(vel_mag_squared).item() / (0.5 * u * u)
        self.kinetic_energy_list.append(kinetic_energy)
        self.timestamp_list.append(timestamp)

        u_x = state.node_data.moments.velocity[0, num_halos:-num_halos, num_halos:-num_halos, 0]
        u_y = state.node_data.moments.velocity[1, num_halos:-num_halos, num_halos:-num_halos, 0]
        dv_dx = (u_y[2:, 1:-1] - u_y[:-2, 1:-1]) / 2.0
        du_dy = (u_x[1:-1, 2:] - u_x[1:-1, :-2]) / 2.0
        vorticity_squared = (dv_dx - du_dy) ** 2
        enstrophy = torch.mean(vorticity_squared).item() * (L * L) / (u * u)
        self.enstrophy_list.append(enstrophy)

        f_i = state.node_data.distributions.old_population[:, num_halos:-num_halos, num_halos:-num_halos, 0]
        # print(f_i.dtype)
        w_i = torch.tensor(state.lattice.lattice_weights()).cuda() 
        entropy = torch.sum(-f_i * torch.log(f_i / w_i.unsqueeze(-1).unsqueeze(-1)), dim=0)
        total_entropy = torch.sum(entropy).item() / (L * L)
        self.entropy_list.append(total_entropy)
        # print(total_entropy.shape)
        # print(a)


    def write_quantities(self, state):
        quantity_folder = self._result_folder.joinpath("quantities")
        file_o.create_folder(quantity_folder)
        # Create numpy arrays for all lists
        kinetic_energy_array = np.array(self.kinetic_energy_list)
        enstrophy_array = np.array(self.enstrophy_list)
        entropy_array = np.array(self.entropy_list)
        timestamp_array = np.array(self.timestamp_list)
        # Save arrays to files
        np.save(quantity_folder.joinpath("kinetic_energy.npy"), kinetic_energy_array)
        np.save(quantity_folder.joinpath("enstrophy.npy"), enstrophy_array)
        np.save(quantity_folder.joinpath("entropy.npy"), entropy_array)
        np.save(quantity_folder.joinpath("timestamp.npy"), timestamp_array)

        # Plot kinetic energy over time
        plt.figure()
        plt.plot(self.timestamp_list, self.kinetic_energy_list, label="Simulated")
        # plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("Kinetic Energy")
        plt.legend()
        plt.grid()
        plt.title("Kinetic Energy Decay")
        plt.savefig(quantity_folder.joinpath("kinetic_energy.png"))
        plt.close()

        # Plot enstrophy over time
        plt.figure()
        plt.plot(self.timestamp_list, self.enstrophy_list, label="Simulated")
        # plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("Enstrophy")
        plt.legend()
        plt.grid()
        plt.title("Enstrophy Evolution")
        plt.savefig(quantity_folder.joinpath("enstrophy.png"))
        plt.close()

        # Plot entropy over time
        plt.figure()
        plt.plot(self.timestamp_list, self.entropy_list, label="Simulated")
        # plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("Entropy")
        plt.legend()
        plt.grid()
        plt.title("Entropy Evolution")
        plt.savefig(quantity_folder.joinpath("entropy.png"))
        plt.close()


def main():

    simulation_setup = TorchlbmSetup("CurrentKBC")
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
    simulation_setup["Output"]["OutputTimeInterval"].value = end_time / 10.0
    simulation_setup["Output"]["Velocity"]["Active"].value = True
    simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 0.04]
    simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = False
    simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["Active"].value = True
    simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["EvaluationTimeInterval"].value = end_time / 100.0

    simulation_setup["Physics"]["MachNumber"].value = u * math.sqrt(3.0)
    simulation_setup["Physics"]["EndTime"].value = end_time
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = u
    simulation_setup["Physics"]["KinematicViscosityPu"].value = nu
    simulation_setup["Physics"]["Precision"].value = "Single"

    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "GNN"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/739087019080992631/59f8b1d89fec49a8be21080a1c19d6db/checkpoints/epoch=99-step=625000.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/671044091917114828/e2af73c363b94dd28fbfb06deefab240/checkpoints/epoch=139-step=875000.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/671044091917114828/13790144cb2c4a9d9d37b7f81ae7d995/checkpoints/epoch=132-step=831250.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/358668294755187420/cd2ec4e3f9d34dfaa1253842fbf530ac/checkpoints/epoch=126-step=793750.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/859728687786415532/2b3c13eda27a4fc8be162e7be57b094f/checkpoints/epoch=135-step=850000.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/859728687786415532/34cbdc162c984a7ca7c5dc1580e87755/checkpoints/epoch=112-step=706250.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/859728687786415532/fab367eeb28d4bdbb061352cf00257dd/checkpoints/epoch=132-step=831250.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/277500390728212834/da21ce2df5454925899cbada09006f6c/checkpoints/epoch=130-step=818750.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/753631913875141340/a40baf362b2b472d8ec19f148ff6f7f6/checkpoints/epoch=139-step=875000.ckpt"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/469983970254995099/8cccdfd1d8e8443a9215d228bae297a1/checkpoints/epoch=139-step=875000.ckpt"

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    check_torchlbm_setup(simulation_setup)

    initial_condition = ShearLayerInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False, output_writer=ShearLayerOutputWriter)
    simulation.run()


if __name__ == "__main__":

    main()

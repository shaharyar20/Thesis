import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.io_tools.output_writer import OutputWriter
import torchlbm.standalone_operations.file_operations as file_o

U = 0.02
diam = 50
Re =  100
L = int(4.1 * diam)
nu = U * diam / Re
end_time = 80 *  diam / U

class VortexStreetInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        return [
            torch.ones_like(X) * U,
            torch.zeros_like(X),
            torch.zeros_like(X),
        ]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X)
    
    def get_bounce_back_mask(self, X, Y, Z):
        mask = torch.zeros_like(X).bool()
        
        x0 = 2 * diam
        y0 = 2 * diam
        r = diam / 2.0
        mask = torch.where(torch.sqrt((X-x0)*(X-x0)+(Y-y0)*(Y-y0)) < r, 1, mask)


        return mask
    
    def get_curve_function(self, X, Y, Z):

        x0 = 2 * diam
        y0 = 2 * diam
        r = diam / 2.0

        circle = lambda x, y, z: torch.sqrt((x - x0)*(x - x0) + (y - y0)*(y - y0)) - r

        return [circle]
    
class VortexStreetOutputWriter(OutputWriter):
    def __init__(self, result_folder: str, logger, state) -> None:
        super().__init__(result_folder, logger, state)
        self.num_lattice_directions = state.lattice.number_of_discrete_velocities()
        self.opposite_directions = state.lattice.opposite_lattice_indices()
        self.lattice_velocities = torch.tensor(state.lattice.lattice_velocities()).int().cuda() # Assuming CUDA, adjust if using CPU
        self.cd_list = []
        self.cl_list = []
        self.timestamp_list = []

    def evaluate_quantities(self, state, timestamp):

        # if timestamp < (60 * diam / U):
        #     return
        fluid_indices = state.boundary_indices
        # print(fluid_indices.shape)
        # print(fluid_indices[10, :])
        force = torch.zeros(3, device=state.node_data.distributions.old_population.device)
        for index in fluid_indices:
            # print(index)
            # force += - self.lattice_velocities[:, self.opposite_directions[index[0]]] * state.node_data.distributions.old_population[
            #     self.opposite_directions[index[0]],
            #     index[1] - self.lattice_velocities[0, index[0]].item(),
            #     index[2] - self.lattice_velocities[1, index[0]].item(),
            #     index[3] - self.lattice_velocities[2, index[0]].item()] - self.lattice_velocities[:, index[0]] * state.node_data.distributions.old_population[
            #     index[0], index[1], index[2], index[3]
            # ]
            # force += self.lattice_velocities[:, index[0]] * (
            #     state.node_data.distributions.old_population[self.opposite_directions[index[0]], index[1], index[2], index[3]] + state.node_data.distributions.old_population[
            #         index[0], index[1] + self.lattice_velocities[0, index[0]].item(), index[2] + self.lattice_velocities[1, index[0]].item(), index[3] + self.lattice_velocities[2, index[0]].item()
            #     ]
            # )
            # force += self.lattice_velocities[:, self.opposite_directions[index[0]]] * (
            #     state.node_data.distributions.old_population[index[0], index[1], index[2], index[3]] + state.node_data.distributions.old_population[
            #         self.opposite_directions[index[0]], index[1] - self.lattice_velocities[0, index[0]].item(), index[2] - self.lattice_velocities[1, index[0]].item(), index[3] - self.lattice_velocities[2, index[0]].item()
            #     ]
            # )
            # print(index)
            direction = self.opposite_directions[index[0]]
            # print(direction, self.opposite_directions[direction], self.lattice_velocities[:, direction], self.lattice_velocities[:, self.opposite_directions[direction]])
            force += self.lattice_velocities[:, direction] * (
                state.node_data.distributions.old_population[direction,
                    index[1] + self.lattice_velocities[0, direction].item(),
                    index[2] + self.lattice_velocities[1, direction].item(),
                    index[3] + self.lattice_velocities[2, direction].item()] + state.node_data.distributions.old_population[
                    self.opposite_directions[direction], index[1], index[2], index[3]
                ])
        # print(a)
        
        cd = 2 * force[0] / (diam * U * U)
        cl = 2 * force[1] / (diam * U * U)
        self.cd_list.append(cd.item())
        self.cl_list.append(cl.item())
        self.timestamp_list.append(timestamp)

        # print(a)
        






        # # All indices
        # fluid_boundary_indices = fluid_indices[:, 1:]
        # # print(fluid_boundary_indices.shape)
        # # print(fluid_boundary_indices[10, :])
        # # Get all unique node indices that are part of the fluid boundary
        # unique_fluid_boundary_indices = torch.unique(fluid_boundary_indices, dim=0)
        # # print(unique_fluid_boundary_indices)
        # # print(unique_fluid_boundary_indices.shape)
        # force = torch.zeros(3, device=state.node_data.distributions.old_population.device)
        # for index in unique_fluid_boundary_indices:
        #     # print(index)
        #     for direction in range(1, self.num_lattice_directions):
        #         # print(direction, self.opposite_directions[direction], self.lattice_velocities[:, direction], self.lattice_velocities[:, self.opposite_directions[direction]])
        #         # # print((index[0] - self.lattice_velocities[0, direction].item(), index[1] - self.lattice_velocities[1, direction].item(), index[2] - self.lattice_velocities[2, direction].item()))
        #         # print(state.node_data.distributions.old_population[direction, index[0], index[1], index[2]])
        #         # print(state.node_data.distributions.old_population[
        #         #     self.opposite_directions[direction], 
        #         #     index[0] - self.lattice_velocities[0, direction].item(), 
        #         #     index[1] - self.lattice_velocities[1, direction].item(),
        #         #     index[2] - self.lattice_velocities[2, direction].item()]
        #         #     # self.opposite_directions[direction], index[0] - self.lattice_velocities[0, direction], index[1] - self.lattice_velocities[1, direction], index[2] - self.lattice_velocities[2, direction]]
        #         # )
        #         # pop = - state.node_data.distributions.old_population[direction, index[0], index[1], index[2]] - state.node_data.distributions.old_population[
        #         #     self.opposite_directions[direction], 
        #         #     index[0] - self.lattice_velocities[0, direction].item(), 
        #         #     index[1] - self.lattice_velocities[1, direction].item(),
        #         #     index[2] - self.lattice_velocities[2, direction].item()]
        #         #     # self.opposite_directions[direction], index[0] - self.lattice_velocities[0, direction], index[1] - self.lattice_velocities[1, direction], index[2] - self.lattice_velocities[2, direction]]
        #         # # print(pop.device, self.lattice_velocities.device, state.node_data.distributions.old_population.device)
        #         # # print(self.lattice_velocities[:, direction] * pop)
        #         # force += self.lattice_velocities[:, direction] * pop
        #         force += (self.lattice_velocities[:, self.opposite_directions[direction]] * state.node_data.distributions.old_population[
        #             self.opposite_directions[direction],
        #             index[0] - self.lattice_velocities[0, direction].item(),
        #             index[1] - self.lattice_velocities[1, direction].item(),
        #             index[2] - self.lattice_velocities[2, direction].item()] - self.lattice_velocities[:, direction] * state.node_data.distributions.old_population[
        #             direction, index[0], index[1], index[2]]
        #         )
                
        # cd = 2 * force[0] / (diam * U * U)
        # cl = 2 * force[1] / (diam * U * U)
        # self.cd_list.append(cd.item())
        # self.cl_list.append(cl.item())
        # self.timestamp_list.append(timestamp)
        # # print(a)
    
    def write_quantities(self, state):
        quantity_folder = self._result_folder.joinpath("quantities")
        file_o.create_folder(quantity_folder)
        # Create numpy arrays for all lists
        cd_array = np.array(self.cd_list)
        cl_array = np.array(self.cl_list)
        timestamp_array = np.array(self.timestamp_list)
        # Save arrays to files
        np.save(quantity_folder.joinpath("cd.npy"), cd_array)
        np.save(quantity_folder.joinpath("timestamp.npy"), timestamp_array)
        np.save(quantity_folder.joinpath("cl.npy"), cl_array)
        # Plot kinetic energy over time
        plt.figure()
        plt.plot(self.timestamp_list, self.cd_list)
        # plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("Drag Coefficient")
        plt.grid()
        plt.title("Drag Coefficient over Time")
        plt.savefig(quantity_folder.joinpath("cd.png"))
        plt.close()
        plt.figure()
        plt.plot(self.timestamp_list, self.cl_list)
        # plt.yscale("log")
        plt.xlabel("Time")
        plt.ylabel("Lift Coefficient")
        plt.grid()
        plt.title("Lift Coefficient over Time")
        plt.savefig(quantity_folder.joinpath("cl.png"))
        plt.close()
        



def main():

    simulation_setup = TorchlbmSetup(f"VortexStreet_Re_{Re}")
    simulation_setup["Domain"]["Dimension"].value = "2D"
    simulation_setup["Domain"]["NodeSize"].value = L
    simulation_setup["Domain"]["CellsPerNode"].value = L
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [4, 1, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Outlet"
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["OutletDensity"].value = 1.0
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "TimeSpaceDependentWall"
    # simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [1.5*U, 0.0, 0.0]
    # simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Equilibrium"
    # simulation_setup["Domain"]["BoundaryConditions"]["West"]["EquilibriumDensity"].value = 1.0
    # simulation_setup["Domain"]["BoundaryConditions"]["West"]["EquilibriumVelocity"].value = [U, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Wall"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["WallVelocity"].value = [0.0, 0.0, 0.0]
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["BounceBackType"].value = "Interpolated"


    simulation_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value = False
    # simulation_setup["InitialCondition"]["Density"].value = "1.0"
    # simulation_setup["InitialCondition"]["Velocity"]["x"].value = inflow_velocity
    # simulation_setup["InitialCondition"]["Velocity"]["y"].value = "0.0"
    # simulation_setup["InitialCondition"]["Velocity"]["z"].value = "0.0"
    # simulation_setup["InitialCondition"]["BounceBackMask"].value = f"lambda x, y, z: torch.where(torch.sqrt((x-0.05)*(x-0.05)+(y-0.025)*(y-0.025)) < {diam/2}, 1, 0)"
    simulation_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value = False
    #simulation_setup["InitialCondition"]["PyTorchFields"]["Density"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/density_0.01196723.pt"
    #simulation_setup["InitialCondition"]["PyTorchFields"]["Velocity"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/velocity_0.01196723.pt"
    

    simulation_setup["Output"]["Active"].value = True
    simulation_setup["Output"]["ModulusArtifactsActive"].value = False
    simulation_setup["Output"]["PrintTimingInformation"].value = False
    simulation_setup["Output"]["OutputTimeInterval"].value = end_time / 20.0
    simulation_setup["Output"]["Velocity"]["Active"].value = True
    simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 2*U]
    simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = False
    simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["Active"].value = True
    simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["ValueBounds"].value = [0.985, 1.015]
    simulation_setup["Output"]["Density"]["UseValueBounds"].value = False
    simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
    simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["EvaluationTimeInterval"].value = end_time / 2000.0

    simulation_setup["Physics"]["MachNumber"].value = U * math.sqrt(3.0)
    simulation_setup["Physics"]["EndTime"].value = end_time
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = U
    simulation_setup["Physics"]["KinematicViscosityPu"].value = nu
    simulation_setup["Physics"]["Precision"].value = "Single"
    # simulation_setup["Physics"]["VolumeForces"]["Active"].value = False
    # simulation_setup["Physics"]["VolumeForces"]["Type"].value = "ShanChen"
    # simulation_setup["Physics"]["VolumeForces"]["ForceVector"].value = [0.0, 0.0, 0.0]
    # simulation_setup["Physics"]["CarreauYasuda"]["Active"].value = False

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "GNN"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/617757830258376539/1b750f15d097483eafe139cdc2216dbb/checkpoints/epoch=198-step=1243750.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "../distribution_learning/mlruns/358668294755187420/cd2ec4e3f9d34dfaa1253842fbf530ac/checkpoints/epoch=126-step=793750.ckpt"
    # simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"
    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
    # simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"

    check_torchlbm_setup(simulation_setup)

    initial_condition = VortexStreetInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False, output_writer=VortexStreetOutputWriter)
    simulation.run()


if __name__ == "__main__":

    main()


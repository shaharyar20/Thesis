import torch
import numpy as np

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup


class PorousMediaFlowInitialCondition(TorchlbmInitialCondition):
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
    
    def get_bounce_back_mask(self, X, Y, Z):
        mask = torch.ones_like(X).bool()

        aneurysm_mask = np.load("/home/jwinter/Test_Meshes/Realistic_Aneu/2D/Resolution_400/slice_0_400.npy").astype(int)
        aneurysm_mask = torch.from_numpy(aneurysm_mask)
        aneurysm_mask = torch.transpose(aneurysm_mask, 1, 0)
        # print(mask.shape)
        mask[1:1+aneurysm_mask.shape[0],3:3+aneurysm_mask.shape[1],0] = 1-aneurysm_mask

        mask = torch.flip(mask, [1])

        mask[0,:] = mask[1,:]
        mask[-1,:] = mask[-2,:]


        return mask ###### Inlet: 14 - 60 Cell size: 0.0025
        ####### Start: 0.035 - 0.15


def main():


    inflow_velocity = 0.2

    cell_size = 2*0.001649/(60-13)
    cells_per_node = 20
    node_size = cells_per_node * cell_size

    density = 1060
    kinematic_viscosity_inf = 0.0035 / density
    kinematic_viscosity_0 = 0.16 / density

    simulation_setup = TorchlbmSetup(f"Aneurysm")
    simulation_setup["Domain"]["Dimension"].value = "2D"
    simulation_setup["Domain"]["NodeSize"].value = node_size
    simulation_setup["Domain"]["CellsPerNode"].value = cells_per_node
    simulation_setup["Domain"]["NumHaloCells"].value = 1
    simulation_setup["Domain"]["NodeRatio"].value = [20, 11, 1]
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Outlet"
    simulation_setup["Domain"]["BoundaryConditions"]["East"]["OutletDensity"].value = 1.0
    simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "TimeSpaceDependentWall"
    simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
    simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

    simulation_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value = False
    simulation_setup["InitialCondition"]["Density"].value = "1.0"
    simulation_setup["InitialCondition"]["Velocity"]["x"].value = 0.0#0.16777750973676994
    simulation_setup["InitialCondition"]["Velocity"]["y"].value = "0.0"
    simulation_setup["InitialCondition"]["Velocity"]["z"].value = "0.0"
    # simulation_setup["InitialCondition"]["BounceBackMask"].value = f"lambda x, y, z: torch.where(torch.sqrt((x-0.1)*(x-0.1)+(y-0.025)*(y-0.025)) < {radius}, 1, 0)"
    simulation_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value = False
    simulation_setup["InitialCondition"]["PyTorchFields"]["Density"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/density_0.01196723.pt"
    simulation_setup["InitialCondition"]["PyTorchFields"]["Velocity"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/velocity_0.01196723.pt"
    

    simulation_setup["Output"]["Active"].value = True
    simulation_setup["Output"]["ModulusArtifactsActive"].value = False
    simulation_setup["Output"]["PrintTimingInformation"].value = False
    simulation_setup["Output"]["OutputTimeInterval"].value = 0.1 #0.2
    simulation_setup["Output"]["Velocity"]["Active"].value = True
    simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 1.5*inflow_velocity]
    simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
    simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["Density"]["Active"].value = True
    simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
    simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
    simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

    simulation_setup["Physics"]["MachNumber"].value = 0.2
    simulation_setup["Physics"]["EndTime"].value = 3.0
    simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 1.5 * inflow_velocity
    simulation_setup["Physics"]["KinematicViscosityPu"].value = kinematic_viscosity_0
    simulation_setup["Physics"]["Precision"].value = "Single"
    simulation_setup["Physics"]["VolumeForces"]["Active"].value = False
    simulation_setup["Physics"]["VolumeForces"]["Type"].value = "ShanChen"
    simulation_setup["Physics"]["VolumeForces"]["ForceVector"].value = [0.0, 0.0, 0.0]

    simulation_setup["Physics"]["CarreauYasuda"]["Active"].value = True
    simulation_setup["Physics"]["CarreauYasuda"]["viscosity_inf"].value = kinematic_viscosity_inf
    simulation_setup["Physics"]["CarreauYasuda"]["viscosity_0"].value = kinematic_viscosity_0
    simulation_setup["Physics"]["CarreauYasuda"]["lam"].value = 8.2
    simulation_setup["Physics"]["CarreauYasuda"]["n"].value = 0.2128
    simulation_setup["Physics"]["CarreauYasuda"]["a"].value = 0.64

    simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
    simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
    simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

    simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "SRT"
    simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"
    simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
    simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"

    check_torchlbm_setup(simulation_setup)

    initial_condition = PorousMediaFlowInitialCondition(simulation_setup)
    simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False)
    simulation.run()


if __name__ == "__main__":

    main()

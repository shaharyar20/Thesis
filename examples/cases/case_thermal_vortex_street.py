import torch
import math

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup


class PorousMediaFlowInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        return [
            torch.ones_like(X) * 0.1,
            torch.zeros_like(X),
            torch.zeros_like(X),
        ]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X)
    
    def get_initial_temperature(self, X, Y, Z):
        return torch.zeros_like(X)
    
    def get_bounce_back_mask(self, X, Y, Z):
        mask = torch.zeros_like(X).bool()
        
        Nx = 256
        Ny = 128
        x0 = Nx*0.2
        y0 = Ny*0.5
        r = 8
        mask = torch.where(torch.sqrt((X-x0)*(X-x0)+(Y-y0)*(Y-y0)) < r, 1, mask)


        return mask


def main():


    # inflow_velocity = 0.02
    kinematic_viscosity = 1.e-6 
    radius = 0.005
    diameter = 2 * radius



    Re_list = [200] #[100, 125, 150, 175, 200]#, 225, 250]

    for Re in Re_list:

        # kinematic_viscosity = inflow_velocity * diameter / Re
        inflow_velocity = kinematic_viscosity * Re / diameter

        # print("kinematic visocity", kinematic_viscosity)
        print("inflow_velocity", inflow_velocity)

        simulation_setup = TorchlbmSetup(f"ThermalVortexStreet_Re_{Re}")
        simulation_setup["Domain"]["Dimension"].value = "2D"
        simulation_setup["Domain"]["NodeSize"].value = 128.0
        simulation_setup["Domain"]["CellsPerNode"].value = 128
        simulation_setup["Domain"]["NumHaloCells"].value = 1
        simulation_setup["Domain"]["NodeRatio"].value = [2, 1, 1]
        simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [0.1, 0.0, 0.0]
        simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
        simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

        simulation_setup["Thermal"]["Active"].value = True
        simulation_setup["Thermal"]["HeatConductivity"].value = 0.1/3.0
        simulation_setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value = 0.0
        simulation_setup["Thermal"]["BounceBackTemperature"].value = 1.0

        # simulation_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value = False
        # simulation_setup["InitialCondition"]["Density"].value = "1.0"
        # simulation_setup["InitialCondition"]["Velocity"]["x"].value = inflow_velocity
        # simulation_setup["InitialCondition"]["Velocity"]["y"].value = "0.0"
        # simulation_setup["InitialCondition"]["Velocity"]["z"].value = "0.0"
        # simulation_setup["InitialCondition"]["BounceBackMask"].value = f"lambda x, y, z: torch.where(torch.sqrt((x-0.05)*(x-0.05)+(y-0.05)*(y-0.05)) < {radius}, 1, 0)"
        # simulation_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value = False
        # simulation_setup["InitialCondition"]["PyTorchFields"]["Density"].value = r"D:\TUM\Semester 4\Master Thesis\Development\torchlbm\cases\VortexStreet_Re_200\pytorch_output\density_0.10000000.pt"
        # simulation_setup["InitialCondition"]["PyTorchFields"]["Velocity"].value = r"D:\TUM\Semester 4\Master Thesis\Development\torchlbm\cases\VortexStreet_Re_200\pytorch_output\velocity_0.10000000.pt"
        # simulation_setup["InitialCondition"]["PyTorchFields"]["BounceBackMask"].value = r"D:\TUM\Semester 4\Master Thesis\Development\torchlbm\cases\VortexStreet_Re_200\pytorch_output\bounce_back_mask_0.10000000.pt"
        

        simulation_setup["Output"]["Active"].value = True
        # simulation_setup["Output"]["ModulusArtifactsActive"].value = False
        # simulation_setup["Output"]["PrintTimingInformation"].value = False
        simulation_setup["Output"]["OutputTimeInterval"].value = 250.0 #0.2
        # simulation_setup["Output"]["Velocity"]["Active"].value = True
        # simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 2*inflow_velocity]
        # simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
        # simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
        # simulation_setup["Output"]["Density"]["Active"].value = True
        # simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
        # simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
        # simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

        simulation_setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
        simulation_setup["Physics"]["EndTime"].value = 5000.0
        simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
        simulation_setup["Physics"]["KinematicViscosityPu"].value = 0.2/3.0
        simulation_setup["Physics"]["Precision"].value = "Single"
        simulation_setup["Physics"]["VolumeForces"]["Active"].value = False
        simulation_setup["Physics"]["VolumeForces"]["Type"].value = "ShanChen"
        simulation_setup["Physics"]["VolumeForces"]["ForceVector"].value = [0.0, 0.0, 0.0]

        simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
        simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
        simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

        simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "SRT"

        check_torchlbm_setup(simulation_setup)

        initial_condition = PorousMediaFlowInitialCondition(simulation_setup)
        simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False)
        simulation.run()


if __name__ == "__main__":

    main()

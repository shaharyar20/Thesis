import torch

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup


class PorousMediaFlowInitialCondition(TorchlbmInitialCondition):
    def __init__(self, torchlbm_setup: TorchlbmSetup) -> None:
        super().__init__(torchlbm_setup)

    def get_initial_velocity(self, X, Y, Z):
        return [
            torch.ones_like(X) * 0.02,
            torch.zeros_like(X),
            torch.zeros_like(X),
        ]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X)
    
    def get_bounce_back_mask(self, X, Y, Z):
        mask = torch.zeros_like(X).bool()
        
        x0 = domain_length_x*0.2
        y0 = domain_length_y*0.5
        r = radius
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

        simulation_setup = TorchlbmSetup(f"DebugVortexStreet_Re_{Re}")
        simulation_setup["Domain"]["Dimension"].value = "2D"
        simulation_setup["Domain"]["NodeSize"].value = 0.1
        simulation_setup["Domain"]["CellsPerNode"].value = 128
        simulation_setup["Domain"]["NumHaloCells"].value = 1
        simulation_setup["Domain"]["NodeRatio"].value = [2, 1, 1]
        simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [inflow_velocity, 0.0, 0.0]
        simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
        simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

        simulation_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value = True
        simulation_setup["InitialCondition"]["Density"].value = "1.0"
        simulation_setup["InitialCondition"]["Velocity"]["x"].value = inflow_velocity
        simulation_setup["InitialCondition"]["Velocity"]["y"].value = "0.0"
        simulation_setup["InitialCondition"]["Velocity"]["z"].value = "0.0"
        simulation_setup["InitialCondition"]["BounceBackMask"].value = f"lambda x, y, z: torch.where(torch.sqrt((x-0.05)*(x-0.05)+(y-0.05)*(y-0.05)) < {radius}, 1, 0)"
        simulation_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value = False
        simulation_setup["InitialCondition"]["PyTorchFields"]["Density"].value = r"D:\TUM\Semester 4\Master Thesis\Development\torchlbm\cases\VortexStreet_Re_200\pytorch_output\density_0.10000000.pt"
        simulation_setup["InitialCondition"]["PyTorchFields"]["Velocity"].value = r"D:\TUM\Semester 4\Master Thesis\Development\torchlbm\cases\VortexStreet_Re_200\pytorch_output\velocity_0.10000000.pt"
        simulation_setup["InitialCondition"]["PyTorchFields"]["BounceBackMask"].value = r"D:\TUM\Semester 4\Master Thesis\Development\torchlbm\cases\VortexStreet_Re_200\pytorch_output\bounce_back_mask_0.10000000.pt"
        

        simulation_setup["Output"]["Active"].value = True
        simulation_setup["Output"]["ModulusArtifactsActive"].value = False
        simulation_setup["Output"]["PrintTimingInformation"].value = False
        simulation_setup["Output"]["OutputTimeInterval"].value = 400 #0.2
        simulation_setup["Output"]["Velocity"]["Active"].value = True
        simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 2*inflow_velocity]
        simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
        simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
        simulation_setup["Output"]["Density"]["Active"].value = True
        simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
        simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
        simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

        simulation_setup["Physics"]["MachNumber"].value = 0.25
        simulation_setup["Physics"]["EndTime"].value = 400.0
        simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 1.5* inflow_velocity
        simulation_setup["Physics"]["KinematicViscosityPu"].value = kinematic_viscosity
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

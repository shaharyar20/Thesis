import torch

from torchlbm.torchlbm import LbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup


class VortexStreetInitialCondition(TorchlbmInitialCondition):
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


    inflow_velocity = 0.02
    # kinematic_viscosity = 1.e-6
    radius = 0.005
    diameter = 2 * radius



    Re_list = [150] #[100, 125, 150, 175, 200]#, 225, 250]

    for Re in Re_list:

        kinematic_viscosity = inflow_velocity * diameter / Re
        # inflow_velocity = kinematic_viscosity * Re / diameter

        print("kinematic visocity", kinematic_viscosity)
        # print("inflow_velocity", inflow_velocity)

        St = 0.198 * (1.0 - 19.7 / Re)
        print(f"Strouhal: {St}")
        frequency = St * inflow_velocity / diameter
        print(f"Frequency: {frequency}")
        period = 1.0 / frequency
        print(f"Period: {period}")

        simulation_setup = TorchlbmSetup(f"VortexStreet_Re_{Re}")
        simulation_setup["Domain"]["Dimension"].value = "2D"
        simulation_setup["Domain"]["NodeSize"].value = 0.05
        simulation_setup["Domain"]["CellsPerNode"].value = 200
        simulation_setup["Domain"]["NumHaloCells"].value = 1
        simulation_setup["Domain"]["NodeRatio"].value = [4, 1, 1]
        simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "Outlet"
        simulation_setup["Domain"]["BoundaryConditions"]["East"]["OutletDensity"].value = 1.0
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [inflow_velocity, 0.0, 0.0]
        simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["North"]["WallVelocity"].value = [0.0, 0.0, 0.0]
        simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["South"]["WallVelocity"].value = [0.0, 0.0, 0.0]
        simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "Periodic"
        simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "Periodic"

        simulation_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value = True
        simulation_setup["InitialCondition"]["Density"].value = "1.0"
        simulation_setup["InitialCondition"]["Velocity"]["x"].value = inflow_velocity
        simulation_setup["InitialCondition"]["Velocity"]["y"].value = "0.0"
        simulation_setup["InitialCondition"]["Velocity"]["z"].value = "0.0"
        simulation_setup["InitialCondition"]["BounceBackMask"].value = f"lambda x, y, z: torch.where(torch.sqrt((x-0.05)*(x-0.05)+(y-0.025)*(y-0.025)) < {radius}, 1, 0)"
        simulation_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value = False
        #simulation_setup["InitialCondition"]["PyTorchFields"]["Density"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/density_0.01196723.pt"
        #simulation_setup["InitialCondition"]["PyTorchFields"]["Velocity"].value = "/home/jwinter/Development/TorchLBM/cases/modulus_bridge/KarmanVortexStreet/pytorch_output/velocity_0.01196723.pt"
        

        simulation_setup["Output"]["Active"].value = True
        simulation_setup["Output"]["ModulusArtifactsActive"].value = False
        simulation_setup["Output"]["PrintTimingInformation"].value = False
        simulation_setup["Output"]["OutputTimeInterval"].value = 2 * period
        simulation_setup["Output"]["Velocity"]["Active"].value = True
        simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 2*inflow_velocity]
        simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
        simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
        simulation_setup["Output"]["Density"]["Active"].value = True
        simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
        simulation_setup["Output"]["Density"]["ValueBounds"].value = [0.985, 1.015]
        simulation_setup["Output"]["Density"]["UseValueBounds"].value = True
        simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
        simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

        simulation_setup["Physics"]["MachNumber"].value = 0.2
        simulation_setup["Physics"]["EndTime"].value = 50 * period
        simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 2* inflow_velocity
        simulation_setup["Physics"]["KinematicViscosityPu"].value = kinematic_viscosity
        simulation_setup["Physics"]["Precision"].value = "Single"
        simulation_setup["Physics"]["VolumeForces"]["Active"].value = False
        simulation_setup["Physics"]["VolumeForces"]["Type"].value = "ShanChen"
        simulation_setup["Physics"]["VolumeForces"]["ForceVector"].value = [0.0, 0.0, 0.0]
        # simulation_setup["Physics"]["CarreauYasuda"]["Active"].value = False

        simulation_setup["Lattice"]["NSE"]["1D"].value = "D1Q2"
        simulation_setup["Lattice"]["NSE"]["2D"].value = "D2Q9"
        simulation_setup["Lattice"]["NSE"]["3D"].value = "D3Q19"

        simulation_setup["Algorithm"]["Operators"]["Collision"]["Type"].value = "SRT"
        simulation_setup["Algorithm"]["Operators"]["Collision"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"
        simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["Type"].value = "Classical"
        simulation_setup["Algorithm"]["Operators"]["EquilibriumCalculation"]["ModelPath"].value = "distribution_learning/models/eq_model.pth"

        check_torchlbm_setup(simulation_setup)

        initial_condition = VortexStreetInitialCondition(simulation_setup)
        simulation = LbmSimulation(simulation_setup, initial_condition, use_modulus=False)
        simulation.run()


if __name__ == "__main__":

    main()


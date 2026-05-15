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
            torch.ones_like(X) * (u_inf),
            torch.zeros_like(X),
            torch.zeros_like(X),
        ]

    def get_initial_density(self, X, Y, Z):
        return torch.ones_like(X)
    
    def get_initial_temperature(self, X, Y, Z):
        return torch.ones_like(X) * (T_inf)
    
    def get_bounce_back_mask(self, X, Y, Z):
        mask = torch.zeros_like(X).bool()
        mask = torch.where(torch.sqrt((X-x0)*(X-x0)+(Y-y0)*(Y-y0)) < (D/2.0), 1, mask)
        #mask = torch.where(torch.sqrt((X-x1)*(X-x1)+(Y-y0)*(Y-y0)) < (D/2.0), 1, mask)
        return mask

D = 30
x0 = 5.0 * D
y0 = 7.5 * D
x1 = 8.0 * D
Ma_inf = 1.35
T_inf = 0.25
gamma = 1.4
u_inf = Ma_inf * math.sqrt(gamma * T_inf)
rho_inf = 1.0
Re = 300.0
mu_inf = rho_inf * u_inf * D / Re
Pr = 0.71
Cv = 2.5 #1.0 / (gamma - 1.0)
Cp = 3.5 #Cv + 1
k = mu_inf * Cp / Pr
shifted_velx = 0.5*u_inf #0.5*u_inf
shifted_vely = 0.0
print(f"u_inf: {u_inf}, mu_inf: {mu_inf}, k: {k}")
print(f"cp: {Cp}, cv: {Cv}")
print(f"E_inf: {Cv * T_inf + 0.5 * u_inf * u_inf}")

def main():

        simulation_setup = TorchlbmSetup(f"SupersonicCylinder")
        simulation_setup["Domain"]["Dimension"].value = "2D"
        simulation_setup["Domain"]["NodeSize"].value = 15.0 * D
        simulation_setup["Domain"]["CellsPerNode"].value = 15 * D
        simulation_setup["Domain"]["NumHaloCells"].value = 1
        simulation_setup["Domain"]["NodeRatio"].value = [1, 1, 1]
        simulation_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value = "Wall"
        simulation_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value = [u_inf, 0.0, 0.0]
        simulation_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value = "ZeroGradient"
        simulation_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value = "ZeroGradient"

        simulation_setup["Thermal"]["Active"].value = True
        simulation_setup["Thermal"]["ThermalConductivity"].value = k
        simulation_setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value = T_inf #T_inf * Cv + 0.5 * u_inf * u_inf
        # simulation_setup["Thermal"]["BounceBackTemperature"].value = 0.0 #T_inf * Cv + 0.5 * u_inf * u_inf
        simulation_setup["Thermal"]["Cp"].value = Cp
        simulation_setup["Thermal"]["Cv"].value = Cv
        simulation_setup["Thermal"]["Viscosity"].value = mu_inf
        simulation_setup["Thermal"]["ShiftedVelocityX"].value = shifted_velx
        simulation_setup["Thermal"]["ShiftedVelocityY"].value = shifted_vely

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
        simulation_setup["Output"]["OutputTimeInterval"].value = 30.0 #0.2
        simulation_setup["Output"]["OutputEveryStep"].value = False
        simulation_setup["Output"]["Velocity"]["Active"].value = True
        #simulation_setup["Output"]["Velocity"]["ValueBounds"].value = [0.0, 2*inflow_velocity]
        simulation_setup["Output"]["Velocity"]["UseValueBounds"].value = True
        simulation_setup["Output"]["Velocity"]["Types"].value = ["PyTorch", "Picture"]
        simulation_setup["Output"]["Density"]["Active"].value = True
        simulation_setup["Output"]["Density"]["Types"].value = ["PyTorch", "Picture"]
        simulation_setup["Output"]["BounceBackMask"]["Active"].value = True
        simulation_setup["Output"]["BounceBackMask"]["Types"].value = ["PyTorch", "Picture"]

        simulation_setup["Physics"]["MachNumber"].value = 0.1 * math.sqrt(3.0)
        simulation_setup["Physics"]["EndTime"].value = 900.0
        simulation_setup["Physics"]["CharacteristicVelocityPu"].value = 0.1
        simulation_setup["Physics"]["KinematicViscosityPu"].value = 0.5/3.0
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

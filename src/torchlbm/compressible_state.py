import torch
from typing import List

# from torchlbm.core.collision_models.linear_bgk import EquilibriumCalculationModule
# from torchlbm.core.collision_models.thermal_linear_bgk import ThermalEquilibriumCalculationModule
from torchlbm.core.collision_models.compressible_eq import CompressibleEquilibriumCalculationModule
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup
from torchlbm.exceptions import TorchlbmError
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.core.lattices.lattice_dictionaries import (
    OneDimensionalLattices,
    TwoDimensionalLattices,
    ThreeDimensionalLattices,
)
# from torchlbm.node_data import NodeData, Distributions, Moments
# from torchlbm.thermal_node_data import ThermalNodeData, Distributions, Moments
from torchlbm.compressible_node_data import CompressibleNodeData, Distributions, Moments
from torchlbm.unit_converter import UnitConverter
from torchlbm.logger import Logger


def get_meshgrid_for_node(number_nodes: List[int], lattice_distance: float, cells_per_node: int, num_halo_cells: int, dimension: int) -> List[torch.Tensor]:
    """Returns the meshgrid (X, Y, Z) for a node with a pre-defined origin.

    Args:
        number_nodes (List[int]): The number of nodes in each spatial dimension.
        lattice_distance (float): The lattice distance between two neighboring lattices.
        cells_per_node (int): The number of internal cells per node. The number is the same in each spatial dimension.
        num_halo_cells (int): The number of halo cells.
        dimension (int): The spatial dimension as integer.

    Returns:
        List[torch.Tensor]: The List consisting of the meshgrid tensors. Each tensor has the shape (Tx, Ty, Tz),
                            where Tx, Ty, and Tz denote the total number of cells in each spatial dimension.
    """
    linspace_x = torch.linspace(
        0.0 - (num_halo_cells - 0.5) * lattice_distance,
        0.0 + (num_halo_cells + cells_per_node * number_nodes[0] - 0.5) * lattice_distance,
        cells_per_node * number_nodes[0] + 2 * num_halo_cells,
    )
    linspace_y = (
        torch.linspace(
            0.0 - (num_halo_cells - 0.5) * lattice_distance,
            0.0 + (num_halo_cells + cells_per_node * number_nodes[1] - 0.5) * lattice_distance,
            cells_per_node * number_nodes[1] + 2 * num_halo_cells,
        )
        if dimension != 1
        else torch.tensor([0])
    )
    linspace_z = (
        torch.linspace(
            0.0 - (num_halo_cells - 0.5) * lattice_distance,
            0.0 + (num_halo_cells + cells_per_node * number_nodes[2] - 0.5) * lattice_distance,
            cells_per_node * number_nodes[2] + 2 * num_halo_cells,
        )
        if dimension == 3
        else torch.tensor([0.0])
    )
    [X_meshgrid, Y_meshgrid, Z_meshgrid] = torch.meshgrid([linspace_x, linspace_y, linspace_z], indexing="ij")
    return [X_meshgrid, Y_meshgrid, Z_meshgrid]


def read_initial_condition_from_string(evaluation_string, buffer: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    try:
        float(evaluation_string)
        return torch.ones_like(buffer) * float(evaluation_string)
    except ValueError:
        return eval(evaluation_string)(x, y, z)


class CompressibleTorchlbmState:
    """The state of the simulation that contains all relevant information and also storage-intense field or immersed-boundary data."""

    def __init__(self, torchlbm_setup: TorchlbmSetup, initial_condition: TorchlbmInitialCondition, logger: Logger) -> None:
        """The initializer for the simulation state.

        Args:
            torchlbm_setup (TorchlbmSetup): The setup that specifies all relevant characteristics of the testcase that has to be simulated.
            initial_condition (TorchlbmInitialCondition): The object that contains information about the initial condition.
            logger (Logger): Logger object to write to terminal or log file.
        """
        self.torchlbm_setup = torchlbm_setup
        dimension = self.torchlbm_setup["Domain"]["DimensionInteger"].value

        if not self.torchlbm_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value and initial_condition is None:
            raise TorchlbmError("Initial condition is not read from yaml file, but no initial condition object is provided!")

        self.lattice = ThreeDimensionalLattices[self.torchlbm_setup["Lattice"]["NSE"]["3D"].value]
        if dimension == 2:
            self.lattice = TwoDimensionalLattices[self.torchlbm_setup["Lattice"]["NSE"]["2D"].value]
        if dimension == 1:
            self.lattice = OneDimensionalLattices[self.torchlbm_setup["Lattice"]["NSE"]["1D"].value]

        total_i = self.torchlbm_setup["Domain"]["NodeRatio"].value[0]
        total_j = self.torchlbm_setup["Domain"]["NodeRatio"].value[1] if dimension != 1 else 1
        total_k = self.torchlbm_setup["Domain"]["NodeRatio"].value[2] if dimension == 3 else 1

        node_ratio = [total_i, total_j, total_k]
        node_size = self.torchlbm_setup["Domain"]["NodeSize"].value

        num_halo_cells = self.torchlbm_setup["Domain"]["NumHaloCells"].value
        start = [
            num_halo_cells,
            num_halo_cells if dimension != 1 else 0,
            num_halo_cells if dimension == 3 else 0,
        ]

        end = [
            -num_halo_cells,
            -num_halo_cells if dimension != 1 else 1,
            -num_halo_cells if dimension == 3 else 1,
        ]
        cells_per_node = self.torchlbm_setup["Domain"]["CellsPerNode"].value
        total_cells_per_node = pow(cells_per_node, dimension)
        self.lattice_distance = node_size / cells_per_node

        self.unit_converter = UnitConverter(
            torchlbm_setup=torchlbm_setup,
            logger=logger,
            lattice_distance=self.lattice_distance,
            characteristic_velocity_physical_units=self.torchlbm_setup["Physics"]["CharacteristicVelocityPu"].value,
            kinematic_viscosity=self.torchlbm_setup["Physics"]["KinematicViscosityPu"].value,
            mach_number=self.torchlbm_setup["Physics"]["MachNumber"].value,
        )
        self.torchlbm_setup["Physics"]["CharacteristicVelocityLu"].value = self.unit_converter.convert_velocity_to_lattice_units(
            self.torchlbm_setup["Physics"]["CharacteristicVelocityPu"].value
        )
        self.unit_converter.log_units()

        self.total_number_of_lattices = total_cells_per_node * total_i * total_j * total_k

        self.torchlbm_setup["Physics"]["RelaxationOmega"].value = 1.0 / self.unit_converter.relaxation_parameter_lattice_units

        self.relaxation_omega_temp = 1.0 / (3*self.torchlbm_setup["Thermal"]["ThermalConductivity"].value + 0.5)

        meshgrid_for_node = get_meshgrid_for_node(node_ratio, self.lattice_distance, cells_per_node, num_halo_cells, dimension)

        initial_density = torch.ones_like(meshgrid_for_node[0])
        if self.torchlbm_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value:
            initial_density = read_initial_condition_from_string(
                self.torchlbm_setup["InitialCondition"]["Density"].value, initial_density, meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2]
            )
        elif self.torchlbm_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value:
            density_from_pt = torch.load(self.torchlbm_setup["InitialCondition"]["PyTorchFields"]["Density"].value)
            initial_density[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = density_from_pt
        else:
            initial_density = initial_condition.get_initial_density(meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2])
        initial_density = self.unit_converter.convert_density_to_lattice_units(initial_density)

        initial_velocity_x = torch.zeros_like(meshgrid_for_node[0])
        initial_velocity_y = torch.zeros_like(meshgrid_for_node[0])
        initial_velocity_z = torch.zeros_like(meshgrid_for_node[0])
        if self.torchlbm_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value:
            initial_velocity_x = read_initial_condition_from_string(
                self.torchlbm_setup["InitialCondition"]["Velocity"]["x"].value,
                initial_velocity_x,
                meshgrid_for_node[0],
                meshgrid_for_node[1],
                meshgrid_for_node[2],
            )
            initial_velocity_y = read_initial_condition_from_string(
                self.torchlbm_setup["InitialCondition"]["Velocity"]["y"].value,
                initial_velocity_y,
                meshgrid_for_node[0],
                meshgrid_for_node[1],
                meshgrid_for_node[2],
            )
            initial_velocity_z = read_initial_condition_from_string(
                self.torchlbm_setup["InitialCondition"]["Velocity"]["z"].value,
                initial_velocity_z,
                meshgrid_for_node[0],
                meshgrid_for_node[1],
                meshgrid_for_node[2],
            )
            velocity_profile = torch.cat([initial_velocity_x.unsqueeze(0), initial_velocity_y.unsqueeze(0), initial_velocity_z.unsqueeze(0)], dim=0)
        elif self.torchlbm_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value:
            velocity_profile = torch.cat([initial_velocity_x.unsqueeze(0), initial_velocity_y.unsqueeze(0), initial_velocity_z.unsqueeze(0)], dim=0)
            velocity_from_pt = torch.load(self.torchlbm_setup["InitialCondition"]["PyTorchFields"]["Velocity"].value)
            velocity_profile[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]] = velocity_from_pt
        else:
            [initial_velocity_x, initial_velocity_y, initial_velocity_z] = initial_condition.get_initial_velocity(
                meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2]
            )
            velocity_profile = torch.cat([initial_velocity_x.unsqueeze(0), initial_velocity_y.unsqueeze(0), initial_velocity_z.unsqueeze(0)], dim=0)
        velocity_profile = self.unit_converter.convert_velocity_to_lattice_units(velocity_profile)

        # equilibrium_module = EquilibriumCalculationModule(
        equilibrium_module = CompressibleEquilibriumCalculationModule(
            lattice_velocities=self.lattice.lattice_velocities(),
            lattice_weights=self.lattice.lattice_weights(),
            shifted_velx=self.torchlbm_setup["Thermal"]["ShiftedVelocityX"].value,
            shifted_vely=self.torchlbm_setup["Thermal"]["ShiftedVelocityY"].value,
        )

        initial_bounce_back_mask = torch.empty_like(meshgrid_for_node[0])
        if self.torchlbm_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value:
            initial_bounce_back_mask = read_initial_condition_from_string(
                self.torchlbm_setup["InitialCondition"]["BounceBackMask"].value,
                initial_bounce_back_mask,
                meshgrid_for_node[0],
                meshgrid_for_node[1],
                meshgrid_for_node[2],
            )
        elif self.torchlbm_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value:
            bounce_back_mask_from_pt = torch.load(self.torchlbm_setup["InitialCondition"]["PyTorchFields"]["BounceBackMask"].value)
            initial_bounce_back_mask[start[0] : end[0], start[1] : end[1], start[2] : end[2]] = bounce_back_mask_from_pt
        else:
            initial_bounce_back_mask = initial_condition.get_bounce_back_mask(meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2])

        # Optional signed-distance field for the PonD SDF sub-cell no-slip wall
        # (> 0 fluid, |.| = wall distance). None unless the case's IC provides one.
        self.signed_distance = None
        if not self.torchlbm_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value:
            self.signed_distance = initial_condition.get_signed_distance(
                meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2]
            )

        initial_temperature = initial_condition.get_initial_temperature(meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2])
        initial_energy = 0.5 * (initial_velocity_x**2 + initial_velocity_y**2 + initial_velocity_z**2) + initial_temperature * self.torchlbm_setup["Thermal"]["Cv"].value
        print(initial_energy.shape)

        density_shape = initial_density.shape
        self.node_data: CompressibleNodeData = CompressibleNodeData(
            distributions=Distributions(
                torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
            ),
            moments=Moments(initial_density, velocity_profile, initial_temperature, initial_energy, torch.zeros_like(velocity_profile), torch.zeros_like(velocity_profile)),
            bounce_back_mask=initial_bounce_back_mask.to(torch.int8) if initial_bounce_back_mask is not None else None,
        )
        # The exponential-equilibrium seeding (jacfwd moment solve) is built for the D2Q9
        # moment system. PonD lattices (e.g. D2Q16 = 16 velocities) do not use it -- the
        # PonD driver reseeds the populations with the trivial co-moving equilibria -- so
        # skip it there and leave finite placeholder populations.
        if self.lattice.number_of_discrete_velocities() == 9:
            self.node_data = equilibrium_module(self.node_data)
            self.node_data.distributions.vel_old_population = self.node_data.distributions.vel_new_population
            self.node_data.distributions.temp_old_population = self.node_data.distributions.temp_new_population
        else:
            zeros = torch.zeros_like(self.node_data.distributions.vel_old_population)
            self.node_data.distributions.vel_old_population = zeros.clone()
            self.node_data.distributions.vel_new_population = zeros.clone()
            self.node_data.distributions.temp_old_population = zeros.clone()
            self.node_data.distributions.temp_new_population = zeros.clone()

    def mps(self) -> None:
        """Moves all relevant data to the MPS device (tested for Apple MacBook with M chips.)"""
        self.node_data.mps()

    def cuda(self) -> None:
        """Moves all relevant data to the cuda device."""
        self.node_data.cuda()

    def cpu(self) -> None:
        """Moves all relevant data to the cpu."""
        self.node_data.cpu()

import torch
from typing import List

from torchlbm.core.equilibrium.equilibrium import EquilibriumCalculationModule
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup
from torchlbm.exceptions import TorchlbmError
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.core.lattices.lattice_dictionaries import (
    OneDimensionalLattices,
    TwoDimensionalLattices,
    ThreeDimensionalLattices,
)
from torchlbm.node_data import NodeData, Distributions, Moments
from torchlbm.thermal_node_data import ThermalNodeData, ThermalDistributions, ThermalMoments
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


class TorchlbmState:
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
        self.logger = logger
        self.unit_converter = UnitConverter(
            lattice_distance=self.lattice_distance,
            characteristic_velocity_physical_units=self.torchlbm_setup["Physics"]["CharacteristicVelocityPu"].value,
            kinematic_viscosity=self.torchlbm_setup["Physics"]["KinematicViscosityPu"].value,
            mach_number=self.torchlbm_setup["Physics"]["MachNumber"].value,
        )
        self.torchlbm_setup["Physics"]["CharacteristicVelocityLu"].value = self.unit_converter.convert_velocity_to_lattice_units(
            self.torchlbm_setup["Physics"]["CharacteristicVelocityPu"].value
        )
        self.log_units()

        self.total_number_of_lattices = total_cells_per_node * total_i * total_j * total_k

        meshgrid_for_node = get_meshgrid_for_node(node_ratio, self.lattice_distance, cells_per_node, num_halo_cells, dimension)
        self.west_meshgrid = [meshgrid_for_node[i][:num_halo_cells, :, :] for i in range(3)]
        self.east_meshgrid = [meshgrid_for_node[i][-num_halo_cells:, :, :] for i in range(3)]
        self.south_meshgrid = [meshgrid_for_node[i][:, :num_halo_cells, :] for i in range(3)]
        self.north_meshgrid = [meshgrid_for_node[i][:, -num_halo_cells:, :] for i in range(3)]
        self.bottom_meshgrid = [meshgrid_for_node[i][:, :, :num_halo_cells] for i in range(3)]
        self.top_meshgrid = [meshgrid_for_node[i][:, :, -num_halo_cells:] for i in range(3)]

        self.torchlbm_setup["Physics"]["RelaxationOmega"].value = 1.0 / self.unit_converter.relaxation_parameter_lattice_units
        if self.torchlbm_setup["Physics"]["NonNewtonian"]["Active"].value:
            initial_relaxation_omega = torch.ones_like(meshgrid_for_node[0]) * self.torchlbm_setup["Physics"]["RelaxationOmega"].value
        else:
            initial_relaxation_omega = torch.tensor(self.torchlbm_setup["Physics"]["RelaxationOmega"].value)

        initial_density = torch.ones_like(meshgrid_for_node[0])
        if self.torchlbm_setup["InitialCondition"]["ReadInitialConditionFromYaml"].value:
            initial_density = read_initial_condition_from_string(
                self.torchlbm_setup["InitialCondition"]["Density"].value, initial_density, meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2]
            )
        elif self.torchlbm_setup["InitialCondition"]["ReadInitialFieldsFromPyTorchFiles"].value:
            density_from_pt = torch.load(self.torchlbm_setup["InitialCondition"]["PyTorchFields"]["Density"].value)
            initial_density = density_from_pt
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
            velocity_profile = velocity_from_pt
        else:
            [initial_velocity_x, initial_velocity_y, initial_velocity_z] = initial_condition.get_initial_velocity(
                meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2]
            )
            velocity_profile = torch.cat([initial_velocity_x.unsqueeze(0), initial_velocity_y.unsqueeze(0), initial_velocity_z.unsqueeze(0)], dim=0)
        velocity_profile = self.unit_converter.convert_velocity_to_lattice_units(velocity_profile)

        equilibrium_module = EquilibriumCalculationModule(
            self.lattice.lattice_velocities(),
            self.lattice.lattice_weights(),
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

        density_shape = initial_density.shape

        if self.torchlbm_setup["Physics"]["VolumeForces"]["Active"].value or self.torchlbm_setup["Multiphase"]["Active"].value:
            initial_forcing_velocity = torch.zeros_like(velocity_profile)
            initial_volume_force_field = torch.zeros_like(velocity_profile)
            if self.torchlbm_setup["Physics"]["VolumeForces"]["Type"].value == "Guo":
                initial_collision_source_term = torch.zeros([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]])
            else:
                initial_collision_source_term = None
        else:
            initial_forcing_velocity = None
            initial_volume_force_field = None
            initial_collision_source_term = None

        if self.torchlbm_setup["Thermal"]["Active"].value:
            initial_temperature = initial_condition.get_initial_temperature(meshgrid_for_node[0], meshgrid_for_node[1], meshgrid_for_node[2])
            initial_temp_relaxation_omega = torch.tensor(1.0 / (
                3.0 * self.torchlbm_setup["Thermal"]["HeatConductivity"].value + 0.5
            ))

            self.node_data: ThermalNodeData = ThermalNodeData(
                distributions=ThermalDistributions(
                    torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                    torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                    torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                    torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                ),
                moments=ThermalMoments(
                    initial_density,
                    velocity_profile,
                    initial_temperature,
                    initial_forcing_velocity,
                    initial_volume_force_field,
                ),
                vel_relaxation_omega=initial_relaxation_omega,
                temp_relaxation_omega=initial_temp_relaxation_omega,
                bounce_back_mask=initial_bounce_back_mask.to(torch.int8) if initial_bounce_back_mask is not None else None,
            )
            self.node_data.distributions.vel_new_population = equilibrium_module(
                self.node_data.moments.density,
                self.node_data.moments.velocity,
                self.node_data.moments.forcing_velocity,
            )
            self.node_data.distributions.temp_new_population = equilibrium_module(
                self.node_data.moments.temperature,
                self.node_data.moments.velocity,
                self.node_data.moments.forcing_velocity,
            )
            self.node_data.distributions.vel_old_population = self.node_data.distributions.vel_new_population.clone()
            self.node_data.distributions.temp_old_population = self.node_data.distributions.temp_new_population.clone()

        else:
            self.node_data: NodeData = NodeData(
                distributions=Distributions(
                    torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                    torch.empty([self.lattice.n_discrete_velocities, density_shape[0], density_shape[1], density_shape[2]]),
                    initial_collision_source_term,
                ),
                moments=Moments(initial_density, velocity_profile, initial_forcing_velocity, initial_volume_force_field),
                relaxation_omega=initial_relaxation_omega,
                bounce_back_mask=initial_bounce_back_mask.to(torch.int8) if initial_bounce_back_mask is not None else None,
            )
            self.node_data.distributions.new_population = equilibrium_module(
                self.node_data.moments.density,
                self.node_data.moments.velocity,
                self.node_data.moments.forcing_velocity,
            )
            self.node_data.distributions.old_population = self.node_data.distributions.new_population.clone()

    def mps(self) -> None:
        """Moves all relevant data to the MPS device (tested for Apple MacBook with M chips.)"""
        self.node_data.mps()

    def cuda(self) -> None:
        """Moves all relevant data to the cuda device."""
        self.node_data.cuda()

    def cpu(self) -> None:
        """Moves all relevant data to the cpu."""
        self.node_data.cpu()

    def log_units(self):
        """Logs all relevant information of the unit converter."""
        self.logger.write("\n")
        self.logger.star_line_flush()
        self.logger.write("\n")
        self.logger.write("Conversion factors:")
        self.logger.write("\n")
        self.logger.write_table(
            [
                ["Name", "Value"],
                [],
                [
                    "Length",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_length),
                ],
                [],
                [
                    "Velocity",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_velocity),
                ],
                [],
                [
                    "Time",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_time),
                ],
                [],
                [
                    "Acceleration",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_acceleration),
                ],
                [],
                [
                    "Density",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_density),
                ],
                [],
                [
                    "Pressure",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_pressure),
                ],
                [],
                [
                    "Bending modulus",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_bending_modulus),
                ],
                [],
                [
                    "Shear resistance",
                    "{:10.4f}".format(self.unit_converter.conversion_factor_shear_resistance),
                ],
            ]
        )
        self.logger.write("\n")
        self.logger.star_line_flush()
        self.logger.write("\n")

        self.logger.write("Stability considerations:")
        self.logger.write("\n")
        self.logger.write_table(
            [
                ["Name", "Value"],
                [],
                [
                    "Tau lattice units",
                    "{:10.4f}".format(self.unit_converter.relaxation_parameter_lattice_units),
                ],
                [],
                [
                    "Characteristic velocity in lattice units",
                    "{:10.4f}".format(self.unit_converter.convert_velocity_to_lattice_units(self.unit_converter.characteristic_velocity_physical_units)),
                ],
            ]
        )
        self.logger.write("\n")
        self.logger.star_line_flush()
        self.logger.write("\n")

        if self.torchlbm_setup["Physics"]["NonNewtonian"]["Active"].value:
            if self.torchlbm_setup["Physics"]["NonNewtonian"]["Type"].value == "CarreauYasuda":
                self.logger.write("Carreau Yasuda Viscosity parameters:")
                self.logger.write("\n")
                self.logger.write_table(
                    [
                        ["Name", "Physical Value", "Lattice Value"],
                        [],
                        [
                            "viscosity_0",
                            self.torchlbm_setup["Physics"]["KinematicViscosityPu"].value,
                            "{:10.4f}".format(
                                self.unit_converter.convert_kinematic_viscosity_to_relaxation_time_lattice_units(
                                    self.torchlbm_setup["Physics"]["KinematicViscosityPu"].value
                                )
                            ),
                        ],
                        [],
                        [
                            "viscosity_inf",
                            self.torchlbm_setup["Physics"]["NonNewtonian"]["CarreauYasuda"]["ViscosityInf"].value,
                            "{:10.4f}".format(
                                self.unit_converter.convert_kinematic_viscosity_to_relaxation_time_lattice_units(
                                    self.torchlbm_setup["Physics"]["NonNewtonian"]["CarreauYasuda"]["ViscosityInf"].value
                                )
                            ),
                        ],
                    ]
                )
                self.logger.write("\n")
                self.logger.star_line_flush()
                self.logger.write("\n")

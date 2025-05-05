from typing import List
import torch

from torchlbm.state import TorchlbmState
from torchlbm.boundaries.periodic_boundary_update import PeriodicBoundaryUpdate
from torchlbm.boundaries.wall_boundary_update import WallBoundaryUpdate
from torchlbm.boundaries.inlet_boundary_update import InletBoundaryUpdate
from torchlbm.boundaries.outlet_boundary_condition import OutletBoundaryUpdate
from torchlbm.boundaries.bounce_back_boundary_update import BounceBackBoundaryUpdate
from torchlbm.boundaries.zero_gradient_boundary_update import ZeroGradientBoundaryUpdate
from torchlbm.boundaries.time_space_dependent_wall_boundary_update import TimeSpaceDependentWallBoundaryUpdate
from torchlbm.thermal.boundaries.thermal_wall_boundary_update import ThermalWallBoundaryUpdate

from torchlbm.boundaries.boundary_values.time_space_dependent_wall_velocity import TimeSpaceDependentWallVelocity

def get_thermal_boundary_condition_modules(state: TorchlbmState) -> List:

    """Factory function to return a list of boundary condition update objects.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        List: The created list of boundary condition update objects.
"""
    boundary_condition_modules = []
    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Wall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Wall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Wall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Wall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Wall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Wall"
    ):
        boundary_condition_modules.append(get_thermal_wall_boundary_module(state))

    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Inlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Inlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Inlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Inlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Inlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Inlet"
    ):
        boundary_condition_modules.append(get_thermal_inlet_boundary_module(state))

    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Outlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Outlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Outlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Outlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Outlet"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Outlet"
    ):
        boundary_condition_modules.append(get_thermal_outlet_boundary_module(state))

    # Add bounce back boundary condition here later
    if torch.sum(state.node_data.bounce_back_mask) > 0:
        boundary_condition_modules.append(get_thermal_bounce_back_boundary_module(state))

    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "ZeroGradient"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "ZeroGradient"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "ZeroGradient"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "ZeroGradient"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "ZeroGradient"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "ZeroGradient"
    ):
        boundary_condition_modules.append(get_thermal_zero_gradient_boundary_module(state))

    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Periodic"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Periodic"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Periodic"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Periodic"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Periodic"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Periodic"
    ):
        boundary_condition_modules.append(get_thermal_periodic_boundary_module(state))

    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "TimeSpaceDependentWall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "TimeSpaceDependentWall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "TimeSpaceDependentWall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "TimeSpaceDependentWall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "TimeSpaceDependentWall"
        or state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "TimeSpaceDependentWall"
    ):
        boundary_condition_modules.append(get_thermal_time_space_dependent_wall_boundary_module(state))
    
    
    return boundary_condition_modules


def get_thermal_periodic_boundary_module(state: TorchlbmState) -> PeriodicBoundaryUpdate:
    """Factory function to return a periodic boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        PeriodicBoundaryUpdate: The created object.
    """
    num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value

    access_indices: List[List[int]] = [
        [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
        [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
        [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    ]
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Periodic"
        and state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Periodic"
    ):
        is_i_periodic = True
    else:
        is_i_periodic = False
    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Periodic"
        and state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Periodic"
    ):
        is_j_periodic = True if dimension != 1 else False
    else:
        is_j_periodic = False
    if (
        state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Periodic"
        and state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Periodic"
    ):
        is_k_periodic = True if dimension == 3 else False
    else:
        is_k_periodic = False

    return PeriodicBoundaryUpdate(
        num_halo_cells=num_halo_cells,
        access_indices=access_indices,
        dimension=dimension,
        is_i_periodic=is_i_periodic,
        is_j_periodic=is_j_periodic,
        is_k_periodic=is_k_periodic,
    )


def get_thermal_wall_boundary_module(state: TorchlbmState) -> ThermalWallBoundaryUpdate:
    """Factory function to return a wall boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        WallBoundaryUpdate: The created object.
    """
    num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value

    access_indices: List[List[int]] = [
        [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
        [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
        [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    ]
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    is_east_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Wall"
    east_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["East"]["WallTemperature"].value)
    is_west_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Wall"
    west_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value)
    is_north_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Wall"
    north_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["North"]["WallTemperature"].value)
    is_south_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Wall"
    south_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["South"]["WallTemperature"].value)
    is_top_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Wall"
    top_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["Top"]["WallTemperature"].value)
    is_bottom_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Wall"
    bottom_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["Bottom"]["WallTemperature"].value)

    lattice_velocity = torch.tensor(state.lattice.lattice_velocities())
    lattice_weights = torch.tensor(state.lattice.lattice_weights())

    east_velocities = state.lattice.east_velocities()
    west_velocities = state.lattice.west_velocities()
    north_velocities = state.lattice.north_velocities()
    south_velocities = state.lattice.south_velocities()
    top_velocities = state.lattice.top_velocities()
    bottom_velocities = state.lattice.bottom_velocities()
    opposite_lattice_indices = state.lattice.opposite_lattice_indices()

    return ThermalWallBoundaryUpdate(
        is_east_wall=is_east_wall,
        east_wall_temperature=east_wall_temperature,
        is_west_wall=is_west_wall,
        west_wall_temperature=west_wall_temperature,
        is_north_wall=is_north_wall,
        north_wall_temperature=north_wall_temperature,
        is_south_wall=is_south_wall,
        south_wall_temperature=south_wall_temperature,
        is_top_wall=is_top_wall,
        top_wall_temperature=top_wall_temperature,
        is_bottom_wall=is_bottom_wall,
        bottom_wall_temperature=bottom_wall_temperature,
        num_halo_cells=num_halo_cells,
        access_indices=access_indices,
        dimension=dimension,
        lattice_velocity=lattice_velocity,
        lattice_velocity_integers=lattice_velocity.clone().detach().int().tolist(),
        lattice_weights=lattice_weights,
        east_velocities=east_velocities,
        west_velocities=west_velocities,
        north_velocities=north_velocities,
        south_velocities=south_velocities,
        top_velocities=top_velocities,
        bottom_velocities=bottom_velocities,
        opposite_lattice_indices=opposite_lattice_indices,
    )


def get_thermal_time_space_dependent_wall_boundary_module(state: TorchlbmState) -> TimeSpaceDependentWallBoundaryUpdate:
    """Factory function to return a wall boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        TimeSpaceDependentWallBoundaryUpdate: The created object.
    """
    raise NotImplementedError("TimeSpaceDependentWallBoundaryUpdate is not implemented yet.")
    # num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    # internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value

    # access_indices: List[List[int]] = [
    #     [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    # ]
    # dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    # is_east_time_space_dependent_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "TimeSpaceDependentWall"
    # is_west_time_space_dependent_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "TimeSpaceDependentWall"
    # is_north_time_space_dependent_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "TimeSpaceDependentWall"
    # is_south_time_space_dependent_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "TimeSpaceDependentWall"
    # is_top_time_space_dependent_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "TimeSpaceDependentWall"
    # is_bottom_time_space_dependent_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "TimeSpaceDependentWall"

    # lattice_velocity = torch.tensor(state.lattice.lattice_velocities())
    # lattice_weights = torch.tensor(state.lattice.lattice_weights())

    # east_velocities = state.lattice.east_velocities()
    # west_velocities = state.lattice.west_velocities()
    # north_velocities = state.lattice.north_velocities()
    # south_velocities = state.lattice.south_velocities()
    # top_velocities = state.lattice.top_velocities()
    # bottom_velocities = state.lattice.bottom_velocities()
    # opposite_lattice_indices = state.lattice.opposite_lattice_indices()

    # time_space_dependent_wall_velocity = TimeSpaceDependentWallVelocity(
    #     east_meshgrid=torch.cat([state.east_meshgrid[0].unsqueeze(0), state.east_meshgrid[1].unsqueeze(0), state.east_meshgrid[2].unsqueeze(0)], dim=0),
    #     west_meshgrid=torch.cat([state.west_meshgrid[0].unsqueeze(0), state.west_meshgrid[1].unsqueeze(0), state.west_meshgrid[2].unsqueeze(0)], dim=0),
    #     north_meshgrid=torch.cat([state.north_meshgrid[0].unsqueeze(0), state.north_meshgrid[1].unsqueeze(0), state.north_meshgrid[2].unsqueeze(0)], dim=0),
    #     south_meshgrid=torch.cat([state.south_meshgrid[0].unsqueeze(0), state.south_meshgrid[1].unsqueeze(0), state.south_meshgrid[2].unsqueeze(0)], dim=0),
    #     top_meshgrid=torch.cat([state.top_meshgrid[0].unsqueeze(0), state.top_meshgrid[1].unsqueeze(0), state.top_meshgrid[2].unsqueeze(0)], dim=0),
    #     bottom_meshgrid=torch.cat([state.bottom_meshgrid[0].unsqueeze(0), state.bottom_meshgrid[1].unsqueeze(0), state.bottom_meshgrid[2].unsqueeze(0)], dim=0),
    #     unit_converter=state.unit_converter,
    # )

    # return TimeSpaceDependentWallBoundaryUpdate(
    #     is_east_time_space_dependent_wall=is_east_time_space_dependent_wall,
    #     is_west_time_space_dependent_wall=is_west_time_space_dependent_wall,
    #     is_north_time_space_dependent_wall=is_north_time_space_dependent_wall,
    #     is_south_time_space_dependent_wall=is_south_time_space_dependent_wall,
    #     is_top_time_space_dependent_wall=is_top_time_space_dependent_wall,
    #     is_bottom_time_space_dependent_wall=is_bottom_time_space_dependent_wall,
    #     num_halo_cells=num_halo_cells,
    #     access_indices=access_indices,
    #     dimension=dimension,
    #     lattice_velocity=lattice_velocity,
    #     lattice_velocity_integers=lattice_velocity.clone().detach().int().tolist(),
    #     lattice_weights=lattice_weights,
    #     east_velocities=east_velocities,
    #     west_velocities=west_velocities,
    #     north_velocities=north_velocities,
    #     south_velocities=south_velocities,
    #     top_velocities=top_velocities,
    #     bottom_velocities=bottom_velocities,
    #     opposite_lattice_indices=opposite_lattice_indices,
    #     time_space_dependent_wall_velocity=time_space_dependent_wall_velocity,
    # )


def get_thermal_outlet_boundary_module(state: TorchlbmState) -> OutletBoundaryUpdate:
    """Factory function to return a wall boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        OutletBoundaryUpdate: The created object.
    """
    raise NotImplementedError("OutletBoundaryUpdate is not implemented yet.")
    # num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    # internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value

    # access_indices: List[List[int]] = [
    #     [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    # ]
    # dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    # is_east_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Outlet"
    # east_outlet_density = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["OutletDensity"].value
    # is_west_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Outlet"
    # west_outlet_density = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["OutletDensity"].value
    # is_north_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Outlet"
    # north_outlet_density = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["OutletDensity"].value
    # is_south_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Outlet"
    # south_outlet_density = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["OutletDensity"].value
    # is_top_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Outlet"
    # top_outlet_density = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["OutletDensity"].value
    # is_bottom_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Outlet"
    # bottom_outlet_density = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["OutletDensity"].value

    # lattice_velocity = torch.tensor(state.lattice.lattice_velocities())
    # lattice_weights = torch.tensor(state.lattice.lattice_weights())

    # east_velocities = state.lattice.east_velocities()
    # west_velocities = state.lattice.west_velocities()
    # north_velocities = state.lattice.north_velocities()
    # south_velocities = state.lattice.south_velocities()
    # top_velocities = state.lattice.top_velocities()
    # bottom_velocities = state.lattice.bottom_velocities()
    # opposite_lattice_indices = state.lattice.opposite_lattice_indices()

    # return OutletBoundaryUpdate(
    #     is_east_outlet=is_east_outlet,
    #     is_west_outlet=is_west_outlet,
    #     is_north_outlet=is_north_outlet,
    #     is_south_outlet=is_south_outlet,
    #     is_top_outlet=is_top_outlet,
    #     is_bottom_outlet=is_bottom_outlet,
    #     num_halo_cells=num_halo_cells,
    #     access_indices=access_indices,
    #     dimension=dimension,
    #     lattice_velocity=lattice_velocity,
    #     lattice_weights=lattice_weights,
    #     east_velocities=east_velocities,
    #     west_velocities=west_velocities,
    #     north_velocities=north_velocities,
    #     south_velocities=south_velocities,
    #     top_velocities=top_velocities,
    #     bottom_velocities=bottom_velocities,
    #     opposite_lattice_indices=opposite_lattice_indices,
    #     east_outlet_density=east_outlet_density,
    #     west_outlet_density=west_outlet_density,
    #     north_outlet_density=north_outlet_density,
    #     south_outlet_density=south_outlet_density,
    #     top_outlet_density=top_outlet_density,
    #     bottom_outlet_density=bottom_outlet_density,
    # )


def get_thermal_inlet_boundary_module(state: TorchlbmState) -> WallBoundaryUpdate:
    """Factory function to return a wall boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        WallBoundaryUpdate: The created object.
    """
    raise NotImplementedError("InletBoundaryUpdate is not implemented yet.")
    # num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    # internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value

    # access_indices: List[List[int]] = [
    #     [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    # ]
    # dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    # is_east_inlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Inlet"
    # east_inlet_velocity = state.unit_converter.convert_velocity_to_lattice_units(
    #     torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["WallVelocity"].value)
    # )
    # is_west_inlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Inlet"
    # west_inlet_velocity = state.unit_converter.convert_velocity_to_lattice_units(
    #     torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value)
    # )
    # is_north_inlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Inlet"
    # north_inlet_velocity = state.unit_converter.convert_velocity_to_lattice_units(
    #     torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["WallVelocity"].value)
    # )
    # is_south_inlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Inlet"
    # south_inlet_velocity = state.unit_converter.convert_velocity_to_lattice_units(
    #     torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["WallVelocity"].value)
    # )
    # is_top_inlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Inlet"
    # top_inlet_velocity = state.unit_converter.convert_velocity_to_lattice_units(
    #     torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["WallVelocity"].value)
    # )
    # is_bottom_inlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Inlet"
    # bottom_inlet_velocity = state.unit_converter.convert_velocity_to_lattice_units(
    #     torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["WallVelocity"].value)
    # )

    # lattice_velocity = torch.tensor(state.lattice.lattice_velocities())
    # lattice_weights = torch.tensor(state.lattice.lattice_weights())

    # east_velocities = state.lattice.east_velocities()
    # west_velocities = state.lattice.west_velocities()
    # north_velocities = state.lattice.north_velocities()
    # south_velocities = state.lattice.south_velocities()
    # top_velocities = state.lattice.top_velocities()
    # bottom_velocities = state.lattice.bottom_velocities()
    # opposite_lattice_indices = state.lattice.opposite_lattice_indices()

    # return InletBoundaryUpdate(
    #     is_east_inlet=is_east_inlet,
    #     east_inlet_velocity=east_inlet_velocity,
    #     is_west_inlet=is_west_inlet,
    #     west_inlet_velocity=west_inlet_velocity,
    #     is_north_inlet=is_north_inlet,
    #     north_inlet_velocity=north_inlet_velocity,
    #     is_south_inlet=is_south_inlet,
    #     south_inlet_velocity=south_inlet_velocity,
    #     is_top_inlet=is_top_inlet,
    #     top_inlet_velocity=top_inlet_velocity,
    #     is_bottom_inlet=is_bottom_inlet,
    #     bottom_inlet_velocity=bottom_inlet_velocity,
    #     num_halo_cells=num_halo_cells,
    #     access_indices=access_indices,
    #     dimension=dimension,
    #     lattice_velocity=lattice_velocity,
    #     lattice_velocity_integers=lattice_velocity.clone().detach().int().tolist(),
    #     lattice_weights=lattice_weights,
    #     east_velocities=east_velocities,
    #     west_velocities=west_velocities,
    #     north_velocities=north_velocities,
    #     south_velocities=south_velocities,
    #     top_velocities=top_velocities,
    #     bottom_velocities=bottom_velocities,
    #     opposite_lattice_indices=opposite_lattice_indices,
    # )


def get_thermal_zero_gradient_boundary_module(state: TorchlbmState) -> ZeroGradientBoundaryUpdate:
    """Factotry function for a zero-gradient boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        ZeroGradientBoundaryUpdate: The created object.
    """
    raise NotImplementedError("ZeroGradientBoundaryUpdate is not implemented yet.")
    # num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    # internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value

    # access_indices: List[List[int]] = [
    #     [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
    #     [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    # ]
    # dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    # is_east_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "ZeroGradient"
    # is_west_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "ZeroGradient"
    # is_north_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "ZeroGradient"
    # is_south_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "ZeroGradient"
    # is_top_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "ZeroGradient"
    # is_bottom_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "ZeroGradient"

    # lattice_velocity = torch.tensor(state.lattice.lattice_velocities())
    # lattice_weights = torch.tensor(state.lattice.lattice_weights())

    # east_velocities = state.lattice.east_velocities()
    # west_velocities = state.lattice.west_velocities()
    # north_velocities = state.lattice.north_velocities()
    # south_velocities = state.lattice.south_velocities()
    # top_velocities = state.lattice.top_velocities()
    # bottom_velocities = state.lattice.bottom_velocities()
    # opposite_lattice_indices = state.lattice.opposite_lattice_indices()

    # return ZeroGradientBoundaryUpdate(
    #     is_east_zero_gradient=is_east_zero_gradient,
    #     is_west_zero_gradient=is_west_zero_gradient,
    #     is_north_zero_gradient=is_north_zero_gradient,
    #     is_south_zero_gradient=is_south_zero_gradient,
    #     is_top_zero_gradient=is_top_zero_gradient,
    #     is_bottom_zero_gradient=is_bottom_zero_gradient,
    #     num_halo_cells=num_halo_cells,
    #     access_indices=access_indices,
    #     dimension=dimension,
    #     lattice_velocity=lattice_velocity,
    #     lattice_weights=lattice_weights,
    #     east_velocities=east_velocities,
    #     west_velocities=west_velocities,
    #     north_velocities=north_velocities,
    #     south_velocities=south_velocities,
    #     top_velocities=top_velocities,
    #     bottom_velocities=bottom_velocities,
    #     opposite_lattice_indices=opposite_lattice_indices,
    # )


def get_thermal_bounce_back_boundary_module(state: TorchlbmState) -> BounceBackBoundaryUpdate:
    """Factory function for a bounce back boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        BounceBackBoundaryUpdate: The created object.
    """
    raise NotImplementedError("BounceBackBoundaryUpdate is not implemented yet.")
    # return BounceBackBoundaryUpdate(
    #     opposite_lattice_indices=state.lattice.opposite_lattice_indices(),
    # )

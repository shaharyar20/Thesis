from typing import List
import torch

from torchlbm.state import TorchlbmState
from torchlbm.boundaries.periodic_boundary_update import PeriodicBoundaryUpdate
from torchlbm.boundaries.thermal_periodic_boundary_update import ThermalPeriodicBoundaryUpdate
from torchlbm.boundaries.wall_boundary_update import WallBoundaryUpdate
from torchlbm.boundaries.thermal_wall_boundary_update import ThermalWallBoundaryUpdate
from torchlbm.boundaries.compressible_wall_boundary_update import CompressibleWallBoundaryUpdate
from torchlbm.boundaries.outlet_boundary_condition import OutletBoundaryUpdate
from torchlbm.boundaries.bounce_back_boundary_update import BounceBackBoundaryUpdate
from torchlbm.boundaries.thermal_bounce_back_boundary_update import ThermalBounceBackBoundaryUpdate
from torchlbm.boundaries.compressible_bounce_back_boundary_update import CompressibleBounceBackBoundaryUpdate
from torchlbm.boundaries.zero_gradient_boundary_update import ZeroGradientBoundaryUpdate
from torchlbm.boundaries.thermal_zero_gradient_boundary_update import ThermalZeroGradientBoundaryUpdate


def get_periodic_boundary_module(state: TorchlbmState) -> PeriodicBoundaryUpdate:
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

    # return PeriodicBoundaryUpdate(
    return ThermalPeriodicBoundaryUpdate(
        num_halo_cells=num_halo_cells,
        access_indices=access_indices,
        dimension=dimension,
        is_i_periodic=is_i_periodic,
        is_j_periodic=is_j_periodic,
        is_k_periodic=is_k_periodic,
    )


def get_wall_boundary_module(state: TorchlbmState) -> WallBoundaryUpdate:
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
    east_wall_velocity = state.unit_converter.convert_velocity_to_lattice_units(
        torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["WallVelocity"].value)
    )
    east_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["East"]["WallTemperature"].value)
    is_west_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Wall"
    west_wall_velocity = state.unit_converter.convert_velocity_to_lattice_units(
        torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["WallVelocity"].value)
    )
    west_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["West"]["WallTemperature"].value)
    is_north_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Wall"
    north_wall_velocity = state.unit_converter.convert_velocity_to_lattice_units(
        torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["WallVelocity"].value)
    )
    north_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["North"]["WallTemperature"].value)
    is_south_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Wall"
    south_wall_velocity = state.unit_converter.convert_velocity_to_lattice_units(
        torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["WallVelocity"].value)
    )
    south_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["South"]["WallTemperature"].value)
    is_top_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Wall"
    top_wall_velocity = state.unit_converter.convert_velocity_to_lattice_units(
        torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["WallVelocity"].value)
    )
    top_wall_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BoundaryConditions"]["Top"]["WallTemperature"].value)
    is_bottom_wall = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Wall"
    bottom_wall_velocity = state.unit_converter.convert_velocity_to_lattice_units(
        torch.tensor(state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["WallVelocity"].value)
    )
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

    # return WallBoundaryUpdate(
    # return ThermalWallBoundaryUpdate(
    return CompressibleWallBoundaryUpdate(
        is_east_wall=is_east_wall,
        east_wall_velocity=east_wall_velocity,
        east_wall_temperature=east_wall_temperature,
        is_west_wall=is_west_wall,
        west_wall_velocity=west_wall_velocity,
        west_wall_temperature=west_wall_temperature,
        is_north_wall=is_north_wall,
        north_wall_velocity=north_wall_velocity,
        north_wall_temperature=north_wall_temperature,
        is_south_wall=is_south_wall,
        south_wall_velocity=south_wall_velocity,
        south_wall_temperature=south_wall_temperature,
        is_top_wall=is_top_wall,
        top_wall_velocity=top_wall_velocity,
        top_wall_temperature=top_wall_temperature,
        is_bottom_wall=is_bottom_wall,
        bottom_wall_velocity=bottom_wall_velocity,
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
        shifted_velx=state.torchlbm_setup["Thermal"]["ShiftedVelocityX"].value,
        shifted_vely=state.torchlbm_setup["Thermal"]["ShiftedVelocityY"].value,
        cv=state.torchlbm_setup["Thermal"]["Cv"].value,
    )


def get_outlet_boundary_module(state: TorchlbmState) -> OutletBoundaryUpdate:
    """Factory function to return a wall boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        OutletBoundaryUpdate: The created object.
    """
    num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    internal_cells: List[int] = state.torchlbm_setup["Domain"]["InternalCells"].value

    access_indices: List[List[int]] = [
        [0, num_halo_cells, num_halo_cells + internal_cells[0], 2 * num_halo_cells + internal_cells[0]],
        [0, num_halo_cells, num_halo_cells + internal_cells[1], 2 * num_halo_cells + internal_cells[1]],
        [0, num_halo_cells, num_halo_cells + internal_cells[2], 2 * num_halo_cells + internal_cells[2]],
    ]
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    is_east_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "Outlet"
    is_west_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "Outlet"
    is_north_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "Outlet"
    is_south_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "Outlet"
    is_top_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "Outlet"
    is_bottom_outlet = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "Outlet"

    lattice_velocity = torch.tensor(state.lattice.lattice_velocities())
    lattice_weights = torch.tensor(state.lattice.lattice_weights())

    east_velocities = state.lattice.east_velocities()
    west_velocities = state.lattice.west_velocities()
    north_velocities = state.lattice.north_velocities()
    south_velocities = state.lattice.south_velocities()
    top_velocities = state.lattice.top_velocities()
    bottom_velocities = state.lattice.bottom_velocities()
    opposite_lattice_indices = state.lattice.opposite_lattice_indices()

    return OutletBoundaryUpdate(
        is_east_outlet=is_east_outlet,
        is_west_outlet=is_west_outlet,
        is_north_outlet=is_north_outlet,
        is_south_outlet=is_south_outlet,
        is_top_outlet=is_top_outlet,
        is_bottom_outlet=is_bottom_outlet,
        num_halo_cells=num_halo_cells,
        access_indices=access_indices,
        dimension=dimension,
        lattice_velocity=lattice_velocity,
        lattice_weights=lattice_weights,
        east_velocities=east_velocities,
        west_velocities=west_velocities,
        north_velocities=north_velocities,
        south_velocities=south_velocities,
        top_velocities=top_velocities,
        bottom_velocities=bottom_velocities,
        opposite_lattice_indices=opposite_lattice_indices,
    )


def get_zero_gradient_boundary_module(state: TorchlbmState) -> ZeroGradientBoundaryUpdate:
    """Factotry function for a zero-gradient boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        ZeroGradientBoundaryUpdate: The created object.
    """
    num_halo_cells: int = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value

    is_east_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["East"]["Type"].value == "ZeroGradient"
    is_west_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["West"]["Type"].value == "ZeroGradient"
    is_north_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["North"]["Type"].value == "ZeroGradient"
    is_south_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["South"]["Type"].value == "ZeroGradient"
    is_top_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Top"]["Type"].value == "ZeroGradient"
    is_bottom_zero_gradient = state.torchlbm_setup["Domain"]["BoundaryConditions"]["Bottom"]["Type"].value == "ZeroGradient"

    # return ZeroGradientBoundaryUpdate(
    return ThermalZeroGradientBoundaryUpdate(
        is_east_zero_gradient=is_east_zero_gradient,
        is_west_zero_gradient=is_west_zero_gradient,
        is_north_zero_gradient=is_north_zero_gradient,
        is_south_zero_gradient=is_south_zero_gradient,
        is_top_zero_gradient=is_top_zero_gradient,
        is_bottom_zero_gradient=is_bottom_zero_gradient,
        num_halo_cells=num_halo_cells,
        dimension=dimension,
    )


def get_bounce_back_boundary_module(state: TorchlbmState) -> BounceBackBoundaryUpdate:
    """Factory function for a bounce back boundary update object.

    Args:
        state (TorchlbmState): The state that contains all information about the simulation.

    Returns:
        BounceBackBoundaryUpdate: The created object.
    """
    lattice_weights = torch.tensor(state.lattice.lattice_weights())
    bounce_back_temperature = torch.tensor(state.torchlbm_setup["Thermal"]["BounceBackTemperature"].value)
    # return BounceBackBoundaryUpdate(
    # return ThermalBounceBackBoundaryUpdate(
    return CompressibleBounceBackBoundaryUpdate(
        opposite_lattice_indices=state.lattice.opposite_lattice_indices(),
        lattice_weights=lattice_weights,
        bounce_back_temperature=bounce_back_temperature,
    )

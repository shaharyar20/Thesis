import torch
from typing import List


from torchlbm.unit_converter import UnitConverter

from vtk.util import numpy_support
from vtk import vtkImageData
from torchlbm.state import TorchlbmState


def get_single_node_output_data(state: TorchlbmState) -> vtkImageData:
    """Returns the macroscopic fields of a node as VTK data.
    We have chosen an image-based VTK data type due to the regularity of spatial discretization on a single node of the block-based discretization scheme.

    Args:
        node (NodeData): The node data containing all field information.
        unit_converter (UnitConverter): The unit converter to convert from physical to lattice units and vice verse-
        internal_cells (List[int]): The number of internal cells per spatial dimension.
        num_halos (int): Then number of halo cells.
        dimension (int): The spatial dimension.

    Returns:
        vtkImageData: The vtk data that can be written to a file.
    """
    internal_cells = state.torchlbm_setup["Domain"]["InternalCells"].value
    num_halos = state.torchlbm_setup["Domain"]["NumHaloCells"].value
    dimension = state.torchlbm_setup["Domain"]["DimensionInteger"].value
    node = state.node_data
    unit_converter = state.unit_converter
    start = [
        num_halos,
        num_halos if dimension != 1 else 0,
        num_halos if dimension == 3 else 0,
    ]

    end = [
        -num_halos,
        -num_halos if dimension != 1 else 1,
        -num_halos if dimension == 3 else 1,
    ]

    bounce_back_field = node.bounce_back_mask[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    bounce_back_field_original = bounce_back_field

    density = node.moments.density[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    density = unit_converter.convert_density_to_physical_units(density)
    density = torch.where(bounce_back_field > 0, 1.0, density)

    velocity = node.moments.velocity[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    if node.moments.forcing_velocity is not None:
        velocity += node.moments.forcing_velocity[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
    velocity = unit_converter.convert_velocity_to_physical_units(velocity)
    velocity = torch.where(bounce_back_field.unsqueeze(0) > 0, 0.0, velocity)

    if state.torchlbm_setup["Physics"]["VolumeForces"]["Active"].value:
        forcing_velocity = node.moments.forcing_velocity[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
        forcing_velocity = unit_converter.convert_velocity_to_physical_units(forcing_velocity)
        forcing_velocity = torch.where(bounce_back_field.unsqueeze(0) > 0, 0.0, forcing_velocity)

        volume_force_field = node.moments.volume_force_field[:, start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
        volume_force_field = unit_converter.convert_acceleration_to_physical_units(volume_force_field)
        volume_force_field = torch.where(bounce_back_field.unsqueeze(0) > 0, 0.0, volume_force_field)

    bounce_back_field = torch.moveaxis(bounce_back_field, 2, 0)
    bounce_back_field = torch.moveaxis(bounce_back_field, 1, 2)
    bounce_back_field = bounce_back_field.flatten().detach().numpy()

    density = torch.moveaxis(density, 2, 0)
    density = torch.moveaxis(density, 1, 2)
    density = density.flatten().detach().numpy()

    velocity = torch.moveaxis(velocity, 3, 1)
    velocity = torch.moveaxis(velocity, 2, 3)
    velocity = torch.transpose(velocity.flatten(start_dim=1), 0, 1).detach().numpy()

    density_array = numpy_support.numpy_to_vtk(density)
    density_array.SetName("density")

    velocity_array = numpy_support.numpy_to_vtk(velocity)
    velocity_array.SetName("velocity")

    bounce_back_field_array = numpy_support.numpy_to_vtk(bounce_back_field)
    bounce_back_field_array.SetName("bounce_back_field")

    imageData = vtkImageData()
    imageData.SetExtent(
        0,
        internal_cells[0],
        0,
        internal_cells[1],
        0,
        internal_cells[2],
    )
    imageData.SetOrigin(0.0, 0.0, 0.0)
    imageData.GetCellData().AddArray(density_array)
    imageData.GetCellData().AddArray(velocity_array)
    imageData.GetCellData().AddArray(bounce_back_field_array)

    if state.torchlbm_setup["Thermal"]["Active"].value:

        temperature = node.moments.temperature[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
        # temperature = unit_converter.convert_temperature_to_physical_units(temperature)
        temperature = torch.where(bounce_back_field_original > 0, 0.0, temperature)
        temperature = torch.moveaxis(temperature, 2, 0)
        temperature = torch.moveaxis(temperature, 1, 2)
        temperature = temperature.flatten().detach().numpy()
        temperature_array = numpy_support.numpy_to_vtk(temperature)
        temperature_array.SetName("temperature")
        imageData.GetCellData().AddArray(temperature_array)

    if state.torchlbm_setup["Physics"]["NonNewtonian"]["Active"].value:

        relaxation_omega = node.relaxation_omega[start[0] : end[0], start[1] : end[1], start[2] : end[2]].clone().detach()
        relaxation_time = 1.0 / relaxation_omega
        kinematic_viscosity = unit_converter.convert_relaxation_time_to_kinematic_viscosity_physical_units(relaxation_time)
        kinematic_viscosity = torch.where(bounce_back_field_original > 0, 0.0, kinematic_viscosity)
        kinematic_viscosity = torch.moveaxis(kinematic_viscosity, 2, 0)
        kinematic_viscosity = torch.moveaxis(kinematic_viscosity, 1, 2)
        kinematic_viscosity = kinematic_viscosity.flatten().detach().numpy()
        kinematic_viscosity_array = numpy_support.numpy_to_vtk(kinematic_viscosity)
        kinematic_viscosity_array.SetName("kinematic_viscosity")
        imageData.GetCellData().AddArray(kinematic_viscosity_array)

    if state.torchlbm_setup["Physics"]["VolumeForces"]["Active"].value:

        forcing_velocity = torch.moveaxis(forcing_velocity, 3, 1)
        forcing_velocity = torch.moveaxis(forcing_velocity, 2, 3)
        forcing_velocity = torch.transpose(forcing_velocity.flatten(start_dim=1), 0, 1).detach().numpy()

        volume_force_field = torch.moveaxis(volume_force_field, 3, 1)
        volume_force_field = torch.moveaxis(volume_force_field, 2, 3)
        volume_force_field = torch.transpose(volume_force_field.flatten(start_dim=1), 0, 1).detach().numpy()

        forcing_velocity_array = numpy_support.numpy_to_vtk(forcing_velocity)
        forcing_velocity_array.SetName("forcing_velocity")

        volume_force_field_array = numpy_support.numpy_to_vtk(volume_force_field)
        volume_force_field_array.SetName("volume_force_field")

        imageData.GetCellData().AddArray(forcing_velocity_array)
        imageData.GetCellData().AddArray(volume_force_field_array)



    return imageData

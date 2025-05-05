import torch
import torch.nn as nn

from torchlbm.thermal_node_data import ThermalNodeData

import torch

def get_tensor_memory(tensor):
    """Returns the memory consumption of a tensor in bytes."""
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()
    return 0  # Not a tensor

def compute_memory(obj, seen=None):
    """Recursively computes the total memory usage of all tensors in an object."""
    if seen is None:
        seen = set()
    
    total_memory = 0
    if id(obj) in seen:  # Avoid infinite recursion for circular references
        return 0
    seen.add(id(obj))

    if isinstance(obj, torch.Tensor):
        total_memory += get_tensor_memory(obj)

    elif isinstance(obj, dict):  # If obj is a dictionary, check its values
        for v in obj.values():
            total_memory += compute_memory(v, seen)

    elif isinstance(obj, (list, tuple, set)):  # If obj is a collection, check its elements
        for item in obj:
            total_memory += compute_memory(item, seen)

    elif hasattr(obj, "__dict__"):  # If obj is an object with attributes
        for attr in vars(obj).values():
            total_memory += compute_memory(attr, seen)

    return total_memory

def print_memory_usage(node_data):
    total_bytes = compute_memory(node_data)
    print(f"Total memory: {total_bytes / 1024:.2f} KB ({total_bytes / (1024 ** 2):.2f} MB)")

# # Example usage
# # Assuming node_data is your object containing tensors
# print_memory_usage(node_data)



class ThermalAdvanceModule(nn.Module):
    """The advance module that assembles one timestep based on several other PyTorch modules.

    Args:
        nn (nn.Module): The base class. The AdvanceModule is implemented as a PyTorch module.
                        This allows easily composing algorithms based on consecutive modules.
    """

    def __init__(
        self,
        unit_converter,
        collision_module,
        streaming_module,
        macroscopic_module,
        equilibrium_module,
        multiphase_module,
        boundary_condition_modules,
        thermal_boundary_condition_modules,
        forcing_module,
        is_forcing_active,
        non_newtonian_module,
        is_non_newtonian_active,
    ) -> None:
        """The initializer that stores all relevant submodules as member. Also, important constants and usability objects
        are stored as members.

        Args:
            collision_module (_type_): The collision module.
            streaming_module (_type_): The streaming module.
            macroscopic_module (_type_): The module to calculate macroscopic quantities.
            equilibrium_module (_type_): The module to calculate the equilibrium distribution.
            multiphase_module (_type_): The module to perform multiphase simulations.
            periodic_module (_type_): The module to apply periodic boundaries.
            wall_module (_type_): The module to apply wall boundary conditions.
            zero_gradient_module (_type_): The module to apply zero gradient boundary conditions.
            bounce_back_module (_type_): The module to apply the bounce back boundary condition.
            forcing_module (_type_): The module to apply body forces.
        """
        super(ThermalAdvanceModule, self).__init__()
        self.unit_converter = unit_converter
        self.collision_module = collision_module
        self.streaming_module = streaming_module
        self.macroscopic_module = macroscopic_module
        self.equilibrium_module = equilibrium_module
        self.multiphase_module = multiphase_module
        self.boundary_condition_modules = boundary_condition_modules
        self.thermal_boundary_condition_modules = thermal_boundary_condition_modules
        self.forcing_module = forcing_module
        self.is_forcing_active: bool = is_forcing_active
        self.non_newtonian_module = non_newtonian_module
        self.is_non_newtonian_active: bool = is_non_newtonian_active

    def forward(self, node_data: ThermalNodeData) -> ThermalNodeData:
        """Performs the timestep as the forward pass of the modules.

        Args:
            node_data (NodeData): The node data object that contains all storage heavy data for the midcroscopic and macroscopic quantities.
            ibm_meshes (List[IBMmeshData]): A list of immersed-boundary objects. Each element is one object.
                                            Each object contains the storage-intense data.

        Returns:
            Tuple[NodeData, List[IBMmeshData]]: Return the updated node data and immersed-boundary objects.
        """

        if node_data.moments.volume_force_field is not None:
            node_data.moments.volume_force_field = torch.zeros_like(node_data.moments.volume_force_field)

        # node_data.moments.forcing_velocity = self.multiphase_module(node_data)

        # if self.is_forcing_active:
        #     node_data.moments.forcing_velocity, node_data.moments.volume_force_field = self.forcing_module(node_data.moments.volume_force_field, node_data.moments.density)

        node_data.distributions.vel_new_population = self.equilibrium_module(node_data.moments.density, node_data.moments.velocity, node_data.moments.forcing_velocity)
        node_data.distributions.temp_new_population = self.equilibrium_module(node_data.moments.temperature, node_data.moments.velocity, node_data.moments.forcing_velocity)

        # if self.is_non_newtonian_active:
        #     node_data.relaxation_omega = self.non_newtonian_module(node_data.distributions.old_population, node_data.distributions.new_population, node_data.relaxation_omega, node_data.moments.density)

        if node_data.bounce_back_mask is None:
            node_data.distributions.vel_old_population = self.collision_module(node_data.distributions.vel_old_population, node_data.distributions.vel_new_population, node_data.vel_relaxation_omega)
            node_data.distributions.temp_old_population = self.collision_module(node_data.distributions.temp_old_population, node_data.distributions.temp_new_population, node_data.temp_relaxation_omega)
        else:
            node_data.distributions.vel_old_population = torch.where(
                (node_data.bounce_back_mask > 0),
                node_data.distributions.vel_old_population,
                self.collision_module(node_data.distributions.vel_old_population, node_data.distributions.vel_new_population, node_data.vel_relaxation_omega),
            )
            node_data.distributions.temp_old_population = torch.where(
                (node_data.bounce_back_mask > 0),
                node_data.distributions.temp_old_population,
                self.collision_module(node_data.distributions.temp_old_population, node_data.distributions.temp_new_population, node_data.temp_relaxation_omega),
            )

        node_data.distributions.vel_old_population = self.streaming_module(node_data.distributions.vel_old_population)
        node_data.distributions.temp_old_population = self.streaming_module(node_data.distributions.temp_old_population)

        for module in self.boundary_condition_modules:
            node_data.distributions.vel_old_population = module(node_data.distributions.vel_old_population, node_data.moments.density, node_data.moments.velocity, node_data.bounce_back_mask)

        for module in self.thermal_boundary_condition_modules:
            node_data.distributions.temp_old_population = module(node_data.distributions.temp_old_population, node_data.moments.density, node_data.moments.velocity, node_data.bounce_back_mask)

        node_data.moments.density, node_data.moments.velocity = self.macroscopic_module(node_data.distributions.vel_old_population)
        node_data.moments.temperature, _ = self.macroscopic_module(node_data.distributions.temp_old_population)

        # print_memory_usage(node_data)

        return node_data

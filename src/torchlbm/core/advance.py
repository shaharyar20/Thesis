import torch
import torch.nn as nn

from torchlbm.node_data import NodeData


class AdvanceModule(nn.Module):
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
        super(AdvanceModule, self).__init__()
        self.unit_converter = unit_converter
        self.collision_module = collision_module
        self.streaming_module = streaming_module
        self.macroscopic_module = macroscopic_module
        self.equilibrium_module = equilibrium_module
        self.multiphase_module = multiphase_module
        self.boundary_condition_modules = boundary_condition_modules
        self.forcing_module = forcing_module
        self.is_forcing_active: bool = is_forcing_active
        self.non_newtonian_module = non_newtonian_module
        self.is_non_newtonian_active: bool = is_non_newtonian_active

    def initialize_simulation(self, node_data: NodeData) -> NodeData:

        for i in range(1000):
            print(f"Iteration {i}")

            node_data.moments.volume_force_field = torch.zeros_like(node_data.moments.volume_force_field)

            # node_data.moments.forcing_velocity = self.multiphase_module(node_data)

            if self.is_forcing_active:
                node_data.moments.forcing_velocity = self.forcing_module(node_data)

            node_data = self.equilibrium_module(node_data)

            if self.is_non_newtonian_active:
                node_data.relaxation_omega = self.non_newtonian_module(node_data)

            if node_data.bounce_back_mask is None:
                node_data.distributions.old_population = self.collision_module(node_data)
            else:
                node_data.distributions.old_population = torch.where(
                    (node_data.bounce_back_mask > 0),
                    node_data.distributions.old_population,
                    self.collision_module(node_data),
                )

            node_data = self.streaming_module(node_data)

            for module in self.boundary_condition_modules:
                node_data = module(node_data)

            old_density = node_data.moments.density
            node_data.moments.density = torch.sum(node_data.distributions.old_population, dim=0)
            print(f"Residuum: {torch.linalg.norm(node_data.moments.density - old_density)}")

        return node_data

    def forward(self, node_data: NodeData) -> NodeData:
        """Performs the timestep as the forward pass of the modules.

        Args:
            node_data (NodeData): The node data object that contains all storage heavy data for the midcroscopic and macroscopic quantities.
            ibm_meshes (List[IBMmeshData]): A list of immersed-boundary objects. Each element is one object.
                                            Each object contains the storage-intense data.

        Returns:
            Tuple[NodeData, List[IBMmeshData]]: Return the updated node data and immersed-boundary objects.
        """
        node_data.moments.volume_force_field = torch.zeros_like(node_data.moments.volume_force_field)

        # node_data.moments.forcing_velocity = self.multiphase_module(node_data)

        if self.is_forcing_active:
            node_data.moments.forcing_velocity = self.forcing_module(node_data)

        node_data = self.equilibrium_module(node_data)

        if self.is_non_newtonian_active:
            node_data.relaxation_omega = self.non_newtonian_module(node_data)

        if node_data.bounce_back_mask is None:
            node_data.distributions.old_population = self.collision_module(node_data)
        else:
            node_data.distributions.old_population = torch.where(
                (node_data.bounce_back_mask > 0),
                node_data.distributions.old_population,
                self.collision_module(node_data),
            )

        node_data = self.streaming_module(node_data)

        for module in self.boundary_condition_modules:
            node_data = module(node_data)

        node_data = self.macroscopic_module(node_data)

        return node_data

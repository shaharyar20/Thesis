import torch
import torch.nn as nn
import math

class ShanChenPseudopotentialCalculationModule(nn.Module):
    """PyTorch module that calculates the pseudopotential based on the standard Shan-Chen equation of state.
    It is implemented as a PyTorch module to allow composing algorithms based on consecutively applied modules.

    Args:
        nn (nn.Module): PyTorch module which is used as base class.
    """

    def __init__(
        self,
        ref_density: float,
    ) -> None:
        """Constructor of the module. The constructor is usually called from a factory function.

        Args:
            multiphase_active (bool): Indicate whether multiphase is active.
            solid_density (float): The density of solid nodes prescribed for contact angle.
            bounce_back_mask (torch.Tensor): The mask used for simulating contact angle dynamics. 0 indicates a fluid node,
            1 indicates a solid bounce-back node and 2 indicates a solid wall node.
        """
        super(ShanChenPseudopotentialCalculationModule, self).__init__()
        self.ref_density = ref_density

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """The main functionality of the module as the forward pass of the module.
        It performs the calculation of the pseudopotential forces.

        Args:
            node_data (NodeData): The node data that contains all storage intense field information.

        Returns:
            torch.Tensor: The calculated pseudopotential.
        """
        pseudopotential = self.ref_density * (1.0 - torch.exp(-density / self.ref_density))
        # if self.bounce_back_mask_const is not None:
        #     pseudopotential_solid = self.ref_density * ((1.0 - math.exp(-self.solid_density / self.ref_density)))
        #     pseudopotential = torch.where(
        #         self.bounce_back_mask_const > 0,
        #         pseudopotential_solid,
        #         pseudopotential,
        #     )

        return pseudopotential

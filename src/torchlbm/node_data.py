import typing
import torch


class Moments:
    """The container for the storage-intensive data of the moments of the population
    (which give the macrosopic density and velocity) and several other macroscopic quantities such as forcing velocities for volume forces.
    For vector-based quantities we have chosen a struct-of array based memory layout.
    Thus, a three-dimensional vector (velocity for example) has the layout (3 x Nx x Ny x Nz),
    where 3 denotes the dimension and Nx, Ny, and Nz refer to the number of cells in x-, y-, and z-direction."""

    def __init__(
        self,
        density: torch.Tensor,
        velocity: torch.Tensor,
        forcing_velocity: torch.Tensor,
        volume_force_field: torch.Tensor,
    ) -> None:
        """The constructor for the moment block. For vector-based quantities we have chosen a struct-of array based memory layout.
        Thus, a three-dimensional vector (velocity for example) has the layout (3 x Nx x Ny x Nz),
        where 3 denotes the dimension and Nx, Ny, and Nz refer to the number of cells in x-, y-, and z-direction.

        Args:
            density (torch.Tensor): The density array with shape (Nx x Ny x Nz).
            velocity (torch.Tensor): The velocity array with shape (3 x Nx x Ny x Nz).
            forcing_velocity (torch.Tensor): The forcing velocity array with shape (3 x Nx x Ny x Nz).
            volume_force_field (torch.Tensor): The field array of a volume force with shape (3 x Nx x Ny x Nz).
        """

        self.density = density
        self.velocity = velocity
        self.forcing_velocity = forcing_velocity
        self.volume_force_field = volume_force_field

    def mps(self) -> None:
        """Moves all objects to the mps device."""
        mps_device = torch.device("mps")
        self.density = self.density.to(mps_device)
        self.velocity = self.velocity.to(mps_device)
        self.forcing_velocity = self.forcing_velocity.to(mps_device)
        self.volume_force_field = self.volume_force_field.to(mps_device)

    def cuda(self) -> None:
        """Moves all objects to the cuda device."""
        self.density = self.density.cuda()
        self.velocity = self.velocity.cuda()
        self.forcing_velocity = self.forcing_velocity.cuda()
        self.volume_force_field = self.volume_force_field.cuda()

    def cpu(self) -> None:
        """Moves all objects to the cpu device."""
        self.density = self.density.cpu()
        self.velocity = self.velocity.cpu()
        self.forcing_velocity = self.forcing_velocity.cpu()
        self.volume_force_field = self.volume_force_field.cpu()


class Distributions:
    """A container for the storage intensive data for the distributions characterizing
    the discretized probability-density function of the velocity distribution.
    For vector-based quantities we have chosen a struct-of array based memory layout.
    Thus, a three-dimensional vector (velocity for example) has the layout (3 x Nx x Ny x Nz),
    where 3 denotes the dimension and Nx, Ny, and Nz refer to the number of cells in x-, y-, and z-direction."""

    def __init__(
        self,
        old_population: torch.Tensor,
        new_population: torch.Tensor,
    ) -> None:
        """The constructor for the DistributionBlock. It initializes the populations, i.e. the discretized versions of the velocity distribution.

        Args:
            old_population (torch.Tensor): The old population present at the beginning of a timestep.
            new_population (torch.Tensor): The new population after a timestept.
        """

        self.old_population = old_population
        self.new_population = new_population

    def mps(self) -> None:
        """Moves all objects to the mps device."""
        mps_device = torch.device("mps")
        self.old_population = self.old_population.to(mps_device)
        self.new_population = self.new_population.to(mps_device)

    def cuda(self) -> None:
        """Moves all objects to the cuda device."""
        self.old_population = self.old_population.cuda()
        self.new_population = self.new_population.cuda()

    def cpu(self) -> None:
        """Moves all objects to the cpu device."""
        self.old_population = self.old_population.cpu()
        self.new_population = self.new_population.cpu()


class NodeData:
    """A data container for the storage intensive information about microscopic quantities and macroscopic quantities.
    Contains several other data containers as members.
    It is a pure data container that does not provide any functionality.
    """

    def __init__(
        self,
        distributions: Distributions,
        moments: Moments,
        relaxation_omega: torch.Tensor,
        bounce_back_mask: torch.Tensor,
    ) -> None:
        """The initializer that creates the member for the microscopic and macroscopic quantities.

        Args:
            distributions (Distributions): The data container for the distributions.
            moments (Moments): The data container for the moments.
        """
        self.distributions = distributions
        self.moments = moments
        self.bounce_back_mask = bounce_back_mask
        self.relaxation_omega = relaxation_omega

    def mps(self) -> None:
        """Moves all objects to the mps device."""
        self.distributions.mps()
        self.moments.mps()
        self.relaxation_omega.to("mps")
        self.bounce_back_mask = self.bounce_back_mask.to("mps")

    def cuda(self) -> None:
        """Moves all objects to the cuda device."""
        self.distributions.cuda()
        self.moments.cuda()
        self.relaxation_omega.cuda()
        self.bounce_back_mask = self.bounce_back_mask.cuda()

    def cpu(self) -> None:
        """Moves all objects to the cpu device."""
        self.distributions.cpu()
        self.moments.cpu()
        self.relaxation_omega.cpu()
        self.bounce_back_mask = self.bounce_back_mask.cpu()

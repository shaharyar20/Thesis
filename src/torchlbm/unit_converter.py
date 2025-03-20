import torch

from torchlbm.logger import Logger
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup


class UnitConverter:
    """Provides unit conversions between physical units and lattice units."""

    def __init__(
        self,
        lattice_distance: float,
        characteristic_velocity_physical_units: float,
        kinematic_viscosity: float,
        mach_number: float = 0.05,
    ):
        """Initializes all relevant quantities to perform unit conversion.

        Args:
            torchlbm_setup (TorchlbmSetup): The setup of the simulation that contains all relevant information.
            logger (Logger): The logger to write information to the terminal and the log file.
            lattice_distance (float): The distance between lattices/cells.
            characteristic_velocity_physical_units (float): The characteristiv velocity in physical units.
            kinematic_viscosity (float): The kinematic viscosity in physical units.
            mach_number (float, optional): The mach number to which the problem should be scaled. Should ensure incrompressibility. Defaults to 0.05.
        """
        self.mach_number = mach_number
        self.lattice_distance = lattice_distance
        self.characteristic_velocity_physical_units = characteristic_velocity_physical_units
        self.kinematic_viscosity = kinematic_viscosity
        self.cs = 1 / torch.sqrt(torch.tensor(3.0)).item()

        self.conversion_factor_length = self.lattice_distance
        self.conversion_factor_velocity = self.characteristic_velocity_physical_units / (self.mach_number * self.cs)
        self.conversion_factor_time = self.conversion_factor_length / self.conversion_factor_velocity
        self.conversion_factor_density = 1.0
        self.conversion_factor_acceleration = self.conversion_factor_length / self.conversion_factor_time**2
        self.conversion_factor_pressure = self.conversion_factor_density * self.conversion_factor_velocity**2
        self.conversion_factor_bending_modulus = self.conversion_factor_density * self.conversion_factor_velocity**2 * self.conversion_factor_length**3
        self.conversion_factor_shear_resistance = self.conversion_factor_density * self.conversion_factor_velocity**2 * self.conversion_factor_length
        self.relaxation_parameter_lattice_units = 0.5 + (self.kinematic_viscosity * self.conversion_factor_time) / (
            self.cs**2 * self.conversion_factor_length**2
        )

    def convert_length_to_lattice_units(self, length_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            length_in_physical_units (_type_): The length_in_physical_units

        Returns:
            _type_: The length_in_lattice_units
        """
        return length_in_physical_units / self.conversion_factor_length

    def convert_length_to_physical_units(self, length_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            length_in_lattice_units (_type_): The length_in_lattice_units

        Returns:
            _type_: length_in_physical_units
        """
        return length_in_lattice_units * self.conversion_factor_length

    def convert_velocity_to_lattice_units(self, velocity_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            velocity_in_physical_units (_type_): The velocity_in_physical_units

        Returns:
            _type_: The velocity_in_lattice_units
        """
        return velocity_in_physical_units / self.conversion_factor_velocity

    def convert_velocity_to_physical_units(self, velocity_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            velocity_in_lattice_units (_type_): The velocity_in_lattice_units

        Returns:
            _type_: velocity_in_physical_units
        """
        return velocity_in_lattice_units * self.conversion_factor_velocity

    def convert_time_to_lattice_units(self, time_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            time_in_physical_units (_type_): The time_in_physical_units

        Returns:
            _type_: The time_in_lattice_units
        """
        return time_in_physical_units / self.conversion_factor_time

    def convert_time_to_physical_units(self, time_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            time_in_lattice_units (_type_): The time_in_lattice_units

        Returns:
            _type_: time_in_physical_units
        """
        return time_in_lattice_units * self.conversion_factor_time

    def convert_density_to_lattice_units(self, density_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            density_in_physical_units (_type_): The density_in_physical_units

        Returns:
            _type_: The density_in_lattice_units
        """
        return density_in_physical_units / self.conversion_factor_density

    def convert_density_to_physical_units(self, density_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            density_in_lattice_units (_type_): The density_in_lattice_units

        Returns:
            _type_: density_in_physical_units
        """
        return density_in_lattice_units * self.conversion_factor_density

    def convert_acceleration_to_lattice_units(self, acceleration_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            acceleration_in_physical_units (_type_): The acceleration_in_physical_units

        Returns:
            _type_: The acceleration_in_lattice_units
        """
        return acceleration_in_physical_units / self.conversion_factor_acceleration

    def convert_acceleration_to_physical_units(self, acceleration_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            acceleration_in_lattice_units (_type_): The acceleration_in_lattice_units

        Returns:
            _type_: acceleration_in_physical_units
        """
        return acceleration_in_lattice_units * self.conversion_factor_acceleration

    def convert_pressure_to_lattice_units(self, pressure_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            pressure_in_physical_units (_type_): The pressure_in_physical_units

        Returns:
            _type_: The pressure_in_lattice_units
        """
        return pressure_in_physical_units / self.conversion_factor_pressure

    def convert_pressure_to_physical_units(self, pressure_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            pressure_in_lattice_units (_type_): The pressure_in_lattice_units

        Returns:
            _type_: pressure_in_physical_units
        """
        return pressure_in_lattice_units * self.conversion_factor_pressure

    def convert_bending_modulus_to_lattice_units(self, pressure_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            bending_modulus_in_physical_units (_type_): The bending_modulus_in_physical_units

        Returns:
            _type_: The bending_modulus_in_lattice_units
        """
        return pressure_in_physical_units / self.conversion_factor_bending_modulus

    def convert_bending_modulus_to_physical_units(self, pressure_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            bending_modulus_in_lattice_units (_type_): The bending_modulus_in_lattice_units

        Returns:
            _type_: bending_modulus_in_physical_units
        """
        return pressure_in_lattice_units * self.conversion_factor_bending_modulus

    def convert_shear_resistance_to_lattice_units(self, pressure_in_physical_units):
        """Converts the quantities described in the method name.

        Args:
            shear_resistance_in_physical_units (_type_): The shear_resistance_in_physical_units

        Returns:
            _type_: The shear_resistance_in_lattice_units
        """
        return pressure_in_physical_units / self.conversion_factor_shear_resistance

    def convert_shear_resistance_to_physical_units(self, pressure_in_lattice_units):
        """Converts the quantities described in the method name.

        Args:
            shear_resistance_in_lattice_units (_type_): The shear_resistance_in_lattice_units

        Returns:
            _type_: shear_resistance_in_physical_units
        """
        return pressure_in_lattice_units * self.conversion_factor_shear_resistance

    def convert_kinematic_viscosity_to_relaxation_time_lattice_units(self, kinematic_viscosity_physical_units):
        return 0.5 + (kinematic_viscosity_physical_units * self.conversion_factor_time) / (self.cs**2 * self.conversion_factor_length**2)

    def convert_relaxation_time_to_kinematic_viscosity_physical_units(self, relaxation_time_lattice_units):
        return (relaxation_time_lattice_units - 0.5) * (self.cs**2 * self.conversion_factor_length**2) / self.conversion_factor_time

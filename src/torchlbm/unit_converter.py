import numpy as np

from torchlbm.logger import Logger
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup


class UnitConverter:
    """Provides unit conversions between physical units and lattice units."""

    def __init__(
        self,
        torchlbm_setup: TorchlbmSetup,
        logger: Logger,
        lattice_distance: float,
        characteristic_velocity_physical_units: float,
        kinematic_viscosity: float,
        mach_number=0.05,
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
        self.cs = 1 / np.sqrt(3)

        self.logger: Logger = logger
        self.torchlbm_setup: TorchlbmSetup = torchlbm_setup

    @property
    def conversion_factor_length(self) -> float:
        """Return the conversion factor for length scales.

        Returns:
            float: The conversion factor as float.
        """
        return self.lattice_distance

    @property
    def conversion_factor_velocity(self) -> float:
        """Return the conversion factor for velocity scales.

        Returns:
            float: The conversion factor as float.
        """
        return self.characteristic_velocity_physical_units / (self.mach_number * self.cs)

    @property
    def conversion_factor_time(self) -> float:
        """Return the conversion factor for time scales.

        Returns:
            float: The conversion factor as float.
        """
        return self.conversion_factor_length / self.conversion_factor_velocity

    @property
    def conversion_factor_density(self) -> float:
        """Return the conversion factor for density scales.

        Returns:
            float: The conversion factor as float.
        """
        return 1.0

    @property
    def conversion_factor_acceleration(self) -> float:
        """Return the conversion factor for acceleration scales.

        Returns:
            float: The conversion factor as float.
        """
        return self.conversion_factor_length / self.conversion_factor_time**2

    @property
    def conversion_factor_pressure(self) -> float:
        """Return the conversion factor for pressure scales.

        Returns:
            float: The conversion factor as float.
        """
        return self.conversion_factor_density * self.conversion_factor_velocity**2

    @property
    def conversion_factor_bending_modulus(self) -> float:
        """Return the conversion factor for bending_modulus scales.

        Returns:
            float: The conversion factor as float.
        """
        return self.conversion_factor_density * self.conversion_factor_velocity**2 * self.conversion_factor_length**3

    @property
    def conversion_factor_shear_resistance(self) -> float:
        """Return the conversion factor for shear_resistance scales.

        Returns:
            float: The conversion factor as float.
        """
        return self.conversion_factor_density * self.conversion_factor_velocity**2 * self.conversion_factor_length

    @property
    def relaxation_parameter_lattice_units(self) -> float:
        """Return the relaxation parameter in lattice units.

        Returns:
            float: The relaxation parameter in lattice units.
        """
        return 0.5 + (self.kinematic_viscosity * self.conversion_factor_time) / (self.cs**2 * self.conversion_factor_length**2)

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
                    "{:10.4f}".format(self.conversion_factor_length),
                ],
                [],
                [
                    "Velocity",
                    "{:10.4f}".format(self.conversion_factor_velocity),
                ],
                [],
                [
                    "Time",
                    "{:10.4f}".format(self.conversion_factor_time),
                ],
                [],
                [
                    "Acceleration",
                    "{:10.4f}".format(self.conversion_factor_acceleration),
                ],
                [],
                [
                    "Density",
                    "{:10.4f}".format(self.conversion_factor_density),
                ],
                [],
                [
                    "Pressure",
                    "{:10.4f}".format(self.conversion_factor_pressure),
                ],
                [],
                [
                    "Bending modulus",
                    "{:10.4f}".format(self.conversion_factor_bending_modulus),
                ],
                [],
                [
                    "Shear resistance",
                    "{:10.4f}".format(self.conversion_factor_shear_resistance),
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
                    "{:10.4f}".format(self.relaxation_parameter_lattice_units),
                ],
                [],
                [
                    "Characteristic velocity in lattice units",
                    "{:10.4f}".format(self.convert_velocity_to_lattice_units(self.characteristic_velocity_physical_units)),
                ],
            ]
        )
        self.logger.write("\n")
        self.logger.star_line_flush()
        self.logger.write("\n")

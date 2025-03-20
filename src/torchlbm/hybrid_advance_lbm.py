from pathlib import Path
import math
import torch
import copy

from torchlbm.timer import Timer
from torchlbm.core.advance import AdvanceModule
from torchlbm.state import TorchlbmState
from torchlbm.logger import Logger
import torchlbm.standalone_operations.file_operations as file_o
from torchlbm.io_tools.output_writer import OutputWriter
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup
from torchlbm.setup_definitions.setup_handlers.json_setup_handler import JSONSetupHandler
from torchlbm.setup_definitions.setup_handlers.yaml_setup_handler import YAMLSetupHandler
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.exceptions import TorchlbmError

from torchlbm.module_factory.collision_module_factory import get_collision_module
from torchlbm.module_factory.streaming_module_factory import get_streaming_module
from torchlbm.module_factory.macroscopic_quantitiy_calculation_module_factory import get_macroscopic_quantitiy_calculation_module
from torchlbm.module_factory.boundary_condition_factory import (
    get_periodic_boundary_module,
    get_wall_boundary_module,
    get_outlet_boundary_module,
    get_zero_gradient_boundary_module,
    get_bounce_back_boundary_module,
)
from torchlbm.node_data import NodeData, Distributions, Moments
from torchlbm.module_factory.multiphase_module_factory import get_multiphase_module
from torchlbm.module_factory.forcing_module_factory import get_forcing_module
from torchlbm.module_factory.equilibrium_calculation_module_factory import get_equilibrium_calculation_module
from torchlbm.ml_models.porous_initialization import PorousInitializationModule
from torchlbm.ml_models.operator_advance import OperatorAdvanceModule
from modulus.utils import StaticCaptureEvaluateNoGrad
from modulus.launch.logging import LaunchLogger, initialize_mlflow, initialize_wandb
from tqdm import trange
from functools import wraps
import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time.perf_counter()
        result = f(*args, **kw)
        end_time = time.perf_counter()
        return result, (end_time - start_time)

    return wrap


class HybridAdvancelbmSimulation:
    """The class that performs the timestep loop and contains all relevant information about the simulation."""

    def __init__(self, torchlbm_setup: TorchlbmSetup, initial_condition: TorchlbmInitialCondition, use_modulus: bool = True, use_fno: bool = True) -> None:
        """The initializer for the simulation class.

        Args:
            torchlbm_setup (TorchlbmSetup): The setup that contains all relevant information about the simulation.
            initial_condition (TorchlbmInitialCondition): The object that provides the initial conditions for the simulation.
        """
        simulation_name = torchlbm_setup.name
        self._result_folder = file_o.get_unused_folder(Path(f"./{simulation_name}").absolute())
        file_o.create_folder(self._result_folder)
        self.torchlbm_logger = Logger(
            write_to_file=True,
            log_filename=Path(self._result_folder.joinpath("./torchlbm_log.out")),
            use_modulus_logger=use_modulus,
        )
        self.torchlbm_logger.welcome_message(log_text="Run a LBM simulation")

        self.state: TorchlbmState = TorchlbmState(torchlbm_setup, initial_condition, self.torchlbm_logger)

        json_setup_handler = JSONSetupHandler()
        json_setup_handler.write_to_file(self.state.torchlbm_setup, self._result_folder.joinpath("simulation_setup.json"))
        yaml_setup_handler = YAMLSetupHandler()
        yaml_setup_handler.write_to_file(self.state.torchlbm_setup, self._result_folder.joinpath("simulation_setup.yaml"))

        self.num_cells = torchlbm_setup["Domain"]["TotalCells"].value

        self._output_writer: OutputWriter = OutputWriter(self._result_folder, self.torchlbm_logger, self.state)

        self.timers = {
            "Update": Timer(self.torchlbm_logger, name="Update", indent_level=4),
        }

        if use_modulus:
            from torchlbm.core.modulus_advance import ModulusAdvanceModel

        AdvanceModuleType = ModulusAdvanceModel if use_modulus else AdvanceModule

        self.advance_module = AdvanceModuleType(
            collision_module=get_collision_module(self.state),
            streaming_module=get_streaming_module(self.state),
            macroscopic_module=get_macroscopic_quantitiy_calculation_module(self.state),
            equilibrium_module=get_equilibrium_calculation_module(self.state),
            multiphase_module=get_multiphase_module(self.state),
            periodic_module=get_periodic_boundary_module(self.state),
            wall_module=get_wall_boundary_module(self.state),
            outlet_module=get_outlet_boundary_module(self.state),
            zero_gradient_module=get_zero_gradient_boundary_module(self.state),
            bounce_back_module=get_bounce_back_boundary_module(self.state),
            forcing_module=get_forcing_module(self.state),
            is_forcing_active=self.state.torchlbm_setup["Physics"]["VolumeForces"]["Active"].value,
        )
        print(type(self.advance_module))
        self.advance_module = torch.jit.script(self.advance_module)
        print(self.advance_module)
        self.operator_advance_module = OperatorAdvanceModule(
            unit_converter=self.state.unit_converter,
            num_halos=torchlbm_setup["Domain"]["NumHaloCells"].value,
            dimension=torchlbm_setup["Domain"]["DimensionInteger"].value,
        )
        self.operator_advance_module.eval()
        torch.set_grad_enabled(False)

    def get_output_writer(self) -> OutputWriter:
        """Getter function for the output writer.

        Returns:
            OutputWriter: The output writer that is returned.
        """
        return self._output_writer

    @timing
    def call_advance(self, node_data):
        return self.advance_module(node_data)

    def run(self):
        """Runs the simulation loop."""
        delta_t_pu = self.state.unit_converter.convert_time_to_physical_units(1.0)
        total_number_iterations = math.ceil(self.state.torchlbm_setup["Physics"]["EndTime"].value / delta_t_pu)
        plot_time_interval = self.state.torchlbm_setup["Output"]["OutputTimeInterval"].value

        mlflow_folder = self._output_writer._result_folder.joinpath(f"mlflow_output")
        initialize_mlflow(
            experiment_name="LBM Simulation",
            experiment_desc="Run a LBM simulation",
            run_name=self._result_folder.name,
            run_desc=f"Run the simulation {self._result_folder.name}",
            user_name="jwinter",
            mode="offline",
            tracking_location=str(mlflow_folder),
        )
        LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger

        self.state.unit_converter.log_units()

        reference_node_data = NodeData(
            distributions=Distributions(
                copy.deepcopy(self.state.node_data.distributions.old_population), copy.deepcopy(self.state.node_data.distributions.new_population)
            ),
            moments=Moments(
                density=copy.deepcopy(self.state.node_data.moments.density),
                velocity=copy.deepcopy(self.state.node_data.moments.velocity),
                forcing_velocity=copy.deepcopy(self.state.node_data.moments.forcing_velocity),
                volume_force_field=copy.deepcopy(self.state.node_data.moments.volume_force_field),
            ),
            bounce_back_mask=copy.deepcopy(self.state.node_data.bounce_back_mask),
        )

        if torch.cuda.is_available():
            self.advance_module = self.advance_module.cuda()
            self.operator_advance_module.to("cuda")
            self.state.cuda()
            reference_node_data.cuda()

        if not torch.backends.mps.is_available() and torch.backends.mps.is_built():
            mps_device = torch.device("mps")
            self.advance_module = self.advance_module.to(mps_device)
            self.state.mps()

        error = torch.mean(torch.norm(self.state.node_data.moments.velocity - reference_node_data.moments.velocity))
        self.torchlbm_logger.write(f"Error: {error}")

        # if self.use_fno:
        #     self.state.node_data = self.operator_advance_module(self.state.node_data)

        if self.state.torchlbm_setup["Output"]["Active"].value:
            self._output_writer.write_output(self.state, 0.0)
        if self.state.torchlbm_setup["Output"]["ModulusArtifactsActive"].value:
            with LaunchLogger("Simulation", epoch=0) as modulus_logger:
                artifacts = self._output_writer.get_artifacts(self.state, 0.0)
                for key, value in artifacts.items():
                    modulus_logger.log_figure(value, key)

        error = torch.mean(torch.norm(self.state.node_data.moments.velocity - reference_node_data.moments.velocity))
        self.torchlbm_logger.write(f"Error: {error}")

        error_list = []

        total_number_iterations = 50
        progress_bar = trange(total_number_iterations)
        for iteration_index in progress_bar:
            use_fno = iteration_index % 2 == 1
            # use_fno = True

            if use_fno:
                self.torchlbm_logger.write("Use FNO")
                self.state.node_data = self.operator_advance_module(self.state.node_data)
                self.state.node_data = self.advance_module.inner_model.equilibrium_module(self.state.node_data)
                self.state.node_data.distributions.old_population = self.state.node_data.distributions.new_population
                for j in range(106):
                    reference_node_data, elapsed_time = self.call_advance(reference_node_data)

            else:
                self.torchlbm_logger.write("Use Simulation")
                for j in range(106):
                    reference_node_data, elapsed_time = self.call_advance(reference_node_data)
                    self.state.node_data, elapsed_time = self.call_advance(self.state.node_data)

            self._output_writer.write_output(
                self.state,
                (iteration_index + 1) / (total_number_iterations * 10.0),
            )

            from torch.nn import MSELoss

            velocity_for_loss = self.state.node_data.moments.velocity.detach().clone()
            velocity_for_loss = torch.where(reference_node_data.bounce_back_mask.unsqueeze(0) > 0, 0.0, velocity_for_loss)
            reference_velocity = reference_node_data.moments.velocity.detach().clone()
            reference_velocity = torch.where(reference_node_data.bounce_back_mask.unsqueeze(0) > 0, 0.0, reference_velocity)

            loss = MSELoss()
            error = loss(velocity_for_loss, reference_velocity)
            self.torchlbm_logger.write(f"Error: {error}")
            error_list.append(error.item())

            # if torch.any(torch.isnan(self.state.node_data.distributions.old_population)) or torch.any(
            #     torch.isnan(self.state.node_data.distributions.new_population)
            # ):
            #     self._output_writer.write_output(
            #         self.state,
            #         (iteration_index + 1) / (total_number_iterations * 10.0),
            #     )
            #     raise TorchlbmError("Values in the population tensor are NaN!")

        import pandas as pd

        error_df = pd.DataFrame(error_list)
        error_df.to_csv(str(self._result_folder.joinpath("error.csv")))

        self._output_writer.generate_videos(self.state)

        self.torchlbm_logger.bye_message(log_text="Simulation finished")

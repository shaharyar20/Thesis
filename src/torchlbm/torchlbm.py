from pathlib import Path
import pathlib
import math
import torch

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
from torchlbm.node_data import NodeData

from torchlbm.module_factory.collision_module_factory import get_collision_module
from torchlbm.module_factory.streaming_module_factory import get_streaming_module
from torchlbm.module_factory.macroscopic_quantity_calculation_module_factory import get_macroscopic_quantity_calculation_module
from torchlbm.module_factory.boundary_condition_factory import get_boundary_condition_modules
from torchlbm.module_factory.multiphase_module_factory import get_multiphase_module
from torchlbm.module_factory.forcing_module_factory import get_forcing_module
from torchlbm.module_factory.equilibrium_calculation_module_factory import get_equilibrium_calculation_module
from torchlbm.module_factory.non_newtonian_module_factory import get_non_newtonian_module

from functools import wraps
import time
from tqdm import trange
from contextlib import nullcontext

from torch.profiler import profile, record_function, ProfilerActivity
from torchlbm.utilities.profiling import trace_handler


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start_time = time.perf_counter()
        result = f(*args, **kw)
        end_time = time.perf_counter()
        return result, (end_time - start_time)

    return wrap


class LbmSimulation:
    """The class that performs the timestep loop and contains all relevant information about the simulation."""

    def __init__(self, torchlbm_setup: TorchlbmSetup, initial_condition: TorchlbmInitialCondition, use_modulus: bool = True) -> None:
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

        self.use_modulus = use_modulus
        if self.use_modulus:
            from torchlbm.core.modulus_advance import ModulusAdvanceModel
            from modulus.distributed import DistributedManager

            DistributedManager.initialize()

        AdvanceModuleType = ModulusAdvanceModel if use_modulus else AdvanceModule

        self.advance_module = AdvanceModuleType(
            unit_converter=self.state.unit_converter,
            collision_module=get_collision_module(self.state),
            streaming_module=get_streaming_module(self.state),
            macroscopic_module=get_macroscopic_quantity_calculation_module(self.state),
            equilibrium_module=get_equilibrium_calculation_module(self.state),
            multiphase_module=get_multiphase_module(self.state),
            boundary_condition_modules=get_boundary_condition_modules(self.state),
            forcing_module=get_forcing_module(self.state),
            is_forcing_active=self.state.torchlbm_setup["Physics"]["VolumeForces"]["Active"].value,
            non_newtonian_module=get_non_newtonian_module(self.state),
            is_non_newtonian_active=self.state.torchlbm_setup["Physics"]["NonNewtonian"]["Active"].value,
        )

    def get_output_writer(self) -> OutputWriter:
        """Getter function for the output writer.

        Returns:
            OutputWriter: The output writer that is returned.
        """
        return self._output_writer

    @timing
    def call_advance(self, node_data):
        with torch.no_grad():
            return self.advance_module(node_data)

    def run(self):
        """Runs the simulation loop."""
        delta_t_pu = self.state.unit_converter.convert_time_to_physical_units(1.0)
        total_number_iterations = math.ceil(self.state.torchlbm_setup["Physics"]["EndTime"].value / delta_t_pu)
        plot_time_interval = self.state.torchlbm_setup["Output"]["OutputTimeInterval"].value

        # self.state.unit_converter.log_units()

        if self.use_modulus:
            from modulus.launch.logging import LaunchLogger, initialize_mlflow, initialize_wandb

        if torch.cuda.is_available():
            self.advance_module = self.advance_module.cuda()
            self.state.cuda()

        if not torch.backends.mps.is_available() and torch.backends.mps.is_built():
            mps_device = torch.device("mps")
            self.advance_module = self.advance_module.to(mps_device)
            self.state.mps()
            # self.advance_module.ml_model.model.model.data_handler.to_device(mps_device)

        if self.state.torchlbm_setup["Output"]["Active"].value:
            self._output_writer.write_output(self.state, 0.0)
        if self.state.torchlbm_setup["Output"]["ModulusArtifactsActive"].value and self.use_modulus:
            with LaunchLogger("Simulation", epoch=0) as modulus_logger:

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

                artifacts = self._output_writer.get_artifacts(self.state, 0.0)
                for key, value in artifacts.items():
                    modulus_logger.log_figure(value, key)

        progress_bar = trange(total_number_iterations)

        # self.state.node_data = self.advance_module.initialize_simulation(self.state.node_data)
        # self.advance_module = torch.jit.script(self.advance_module)
        self.advance_module = torch.compile(self.advance_module)

        if self.state.torchlbm_setup["Output"]["ProfilingActive"].value:
            device = "cuda"
            sort_by_keyword = "self_" + device + "_time_total"
            number_wait = int(0.1 * total_number_iterations)
            number_warmup = int(0.1 * total_number_iterations)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=number_wait, warmup=number_warmup, active=total_number_iterations - number_wait - number_warmup),
                on_trace_ready=trace_handler,
                with_stack=True,
                record_shapes=True,
                use_cuda=True,
            ) as profiler:
                for iteration_index in progress_bar:
                    self.state.node_data, elapsed_time = self.call_advance(self.state.node_data)
                    profiler.step()
                print(profiler.key_averages(group_by_stack_n=5).table(sort_by=sort_by_keyword, row_limit=20))
        else:
            for iteration_index in progress_bar:

                self.state.node_data, elapsed_time = self.call_advance(self.state.node_data)

                mlups = self.state.total_number_of_lattices / (1.0e6 * elapsed_time)
                progress_bar.set_postfix(mlups=mlups)

                output_decision_every_step = self.state.torchlbm_setup["Output"]["OutputEveryStep"].value
                current_time_interval_floor = math.floor((iteration_index + 1) * delta_t_pu / plot_time_interval)
                last_time_interval_floor = math.floor(iteration_index * delta_t_pu / plot_time_interval)
                output_decision_interval = current_time_interval_floor != last_time_interval_floor or iteration_index == total_number_iterations - 1
                if (output_decision_every_step or output_decision_interval) and self.state.torchlbm_setup["Output"]["Active"].value:
                    self._output_writer.write_output(
                        self.state,
                        (iteration_index + 1) / (total_number_iterations * 10.0),
                    )
                if (output_decision_every_step or output_decision_interval) and self.state.torchlbm_setup["Output"]["ModulusArtifactsActive"].value:
                    with LaunchLogger("Simulation", epoch=iteration_index) if self.use_modulus else nullcontext() as modulus_logger:
                        artifacts = self._output_writer.get_artifacts(
                            self.state,
                            (iteration_index + 1) / (total_number_iterations * 10.0),
                        )
                        for key, value in artifacts.items():
                            modulus_logger.log_figure(value, key)

                        if self.use_modulus:
                            measured_times = {}
                            measured_times["mlups"] = mlups
                            modulus_logger.log_epoch(measured_times)

                # if torch.any(torch.isinf(self.state.node_data.distributions.old_population)) or torch.any(torch.isinf(self.state.node_data.distributions.new_population)) or torch.any(torch.isinf(self.state.node_data.moments.density)) or torch.any(torch.isinf(self.state.node_data.moments.velocity)):
                #     self._output_writer.write_output(
                #         self.state,
                #         (iteration_index + 1) / (total_number_iterations * 10.0),
                #     )
                #     print("Indices where NaN:")
                #     print(torch.isinf(self.state.node_data.moments.density).nonzero())
                #     print(torch.isinf(self.state.node_data.moments.velocity).nonzero())
                #     print(torch.isinf(self.state.node_data.distributions.old_population).nonzero())
                #     print(torch.isinf(self.state.node_data.distributions.new_population).nonzero())
                #     raise TorchlbmError("Values in the population tensor are NaN!")
            self._output_writer.generate_videos(self.state)

        self.torchlbm_logger.bye_message(log_text="Simulation finished")

import torch

from torchlbm.torchlbm import TorchlbmSimulation
from torchlbm.torchlbm_initial_condition import TorchlbmInitialCondition
from torchlbm.simulation_setup.torchlbm_setup import TorchlbmSetup, check_torchlbm_setup
from torchlbm.setup_definitions.setup_handlers.yaml_setup_handler import YAMLSetupHandler

def main():

    simulation_setup = TorchlbmSetup("KarmanVortexStreet")
    setup_handler = YAMLSetupHandler()
    setup_handler.read_from_file("./yaml_setups/setup_karman.yaml", simulation_setup)

    check_torchlbm_setup(simulation_setup)

    simulation = TorchlbmSimulation(simulation_setup, None)
    simulation.run()


if __name__ == "__main__":

    main()

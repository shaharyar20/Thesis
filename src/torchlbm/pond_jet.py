"""Astrophysical-jet driver: a PonD run with a fixed dense-jet inflow patch.

Same time loop as the base simulation; only the boundaries differ -- open (zero-
gradient) all round, plus a circular west inflow that keeps injecting the jet state.
"""
from torchlbm.pond_lbm import PondLbmSimulation
from torchlbm.core.pond.pond_boundary_adapters import (
    PondZeroGradientBoundaryUpdate,
    PondJetInletBoundaryUpdate,
)


class PondAstrophysicalJetSimulation(PondLbmSimulation):
    """PonD simulation with open boundaries + a steady jet inflow on the west side."""

    def __init__(self, torchlbm_setup, pond_setup, initial_condition,
                 jet_density, jet_velocity, jet_temperature, jet_radius, jet_center=None):
        self._jet_density = float(jet_density)
        self._jet_velocity = list(jet_velocity)
        self._jet_temperature = float(jet_temperature)
        self._jet_radius = float(jet_radius)
        self._jet_center = None if jet_center is None else float(jet_center)
        super().__init__(torchlbm_setup, pond_setup, initial_condition)

    def _build_boundaries(self):
        # Open on all four sides, then overwrite a west patch with the jet inflow.
        eq = self._equilibrium
        nh = self.num_halo_cells
        return [
            PondZeroGradientBoundaryUpdate("west", nh),
            PondZeroGradientBoundaryUpdate("east", nh),
            PondZeroGradientBoundaryUpdate("north", nh),
            PondZeroGradientBoundaryUpdate("south", nh),
            PondJetInletBoundaryUpdate(
                eq, self._jet_density, self._jet_velocity, self._jet_temperature,
                self._jet_radius, nh, jet_center=self._jet_center, side="west",
            ),
        ]

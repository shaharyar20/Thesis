"""Config schema for the PonD solver: every tunable option, its default and range.

Each SetupTag is (name, default, required, converter). The case files set these via
``pond_setup["<Name>"].value = ...``. String options list their allowed values in the
converter.
"""
from torchlbm.setup_definitions.setup_types import SetupSet, SetupTag
from torchlbm.setup_definitions.type_converters import (
    BoolConverter,
    FloatConverter,
    IntConverter,
    StringConverter,
)


class PondSetup(SetupSet):
    """The "Pond" setup group holding all solver knobs."""

    def __init__(self) -> None:
        settings = [
            # Lattice reference temperature T_L: 1.0 for D2Q16, 1/3 for D2Q9T.
            SetupTag("LatticeTemperature", 1.0, False, FloatConverter(0.0, None, False)),
            # Courant number; sets dt each step so max|v_i| dt/dx = cfl (must be < 1).
            SetupTag("CflNumber", 0.2, False, FloatConverter(0.0, 1.0, False)),
            # How the advection target gauge is closed.
            SetupTag("GaugeMode", "interpolated", False,
                     StringConverter(["interpolated", "predictor_corrector"], False)),
            # Neighbour-mean weight for the interpolated gauge (0..1).
            SetupTag("GaugeBlend", 0.5, False, FloatConverter(0.0, 1.0, False)),
            # Immersed-wall treatment (see pond_boundary_adapters).
            SetupTag("WallBc", "noslip", False,
                     StringConverter(["noslip", "sdf_noslip"], False)),
            # Energy closure: "combined" -> gamma = 1.4, "f_only" -> gamma = 2.
            SetupTag("EnergyClosure", "combined", False,
                     StringConverter(["combined", "f_only"], False)),
            # Predictor-corrector: max gauge iterations and convergence tolerances.
            SetupTag("MaxIterations", 2, False, IntConverter(1, None, False)),
            SetupTag("ConvergenceRtol", 1e-5, False, FloatConverter(0.0, None, False)),
            SetupTag("ConvergenceAtol", 1e-8, False, FloatConverter(0.0, None, False)),
            # Zero-guard for the TVD limiter's slope ratio.
            SetupTag("SlopeRatioEpsilon", 1e-10, False, FloatConverter(0.0, None, False)),
            # Blend reconstruction toward 1st-order upwind if a cell would go negative.
            SetupTag("PositivityLimiter", False, False, BoolConverter()),
            # Positivity floors on T and rho (guard 1/rho / sqrt(T) blow-ups).
            SetupTag("TemperatureFloor", 1e-4, False, FloatConverter(0.0, None, False)),
            SetupTag("DensityFloor", 1e-6, False, FloatConverter(0.0, None, False)),
            # Reference density used by inflow / refill boundary conditions.
            SetupTag("InletDensity", 1.0, False, FloatConverter(0.0, None, False)),
            # Write output every N steps (0 = time-interval output only).
            SetupTag("OutputEveryNSteps", 0, False, IntConverter(0, None, False)),
        ]
        super().__init__("Pond", settings)

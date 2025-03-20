from typing import List
import torch
from torch import nn

from torchlbm.unit_converter import UnitConverter
from torchlbm.setup_definitions.enum_utilities import BoundaryLocation


class TimeSpaceDependentWallVelocity(nn.Module):

    def __init__(
        self,
        east_meshgrid: List[torch.Tensor],
        west_meshgrid: List[torch.Tensor],
        north_meshgrid: List[torch.Tensor],
        south_meshgrid: List[torch.Tensor],
        top_meshgrid: List[torch.Tensor],
        bottom_meshgrid: List[torch.Tensor],
        unit_converter: UnitConverter,
    ):
        super(TimeSpaceDependentWallVelocity, self).__init__()

        self.east_meshgrid = east_meshgrid
        self.register_buffer("east_meshgrid_const", self.east_meshgrid)
        self.west_meshgrid = west_meshgrid
        self.register_buffer("west_meshgrid_const", self.west_meshgrid)
        self.north_meshgrid = north_meshgrid
        self.register_buffer("north_meshgrid_const", self.north_meshgrid)
        self.south_meshgrid = south_meshgrid
        self.register_buffer("south_meshgrid_const", self.south_meshgrid)
        self.top_meshgrid = top_meshgrid
        self.register_buffer("top_meshgrid_const", self.top_meshgrid)
        self.bottom_meshgrid = bottom_meshgrid
        self.register_buffer("bottom_meshgrid_const", self.bottom_meshgrid)
        self.unit_converter = unit_converter

        self.time = torch.tensor([0])
        self.register_buffer("time_parameter", self.time)

    def get_east_wall_velocity(self) -> torch.Tensor:
        return torch.cat(
            [torch.zeros_like(self.east_meshgrid_const[0]), torch.zeros_like(self.east_meshgrid_const[0]), torch.zeros_like(self.east_meshgrid_const[0])], dim=0
        )

    def get_west_wall_velocity(self) -> torch.Tensor:
        # Heartbeat Inflow
        """t = self.unit_converter.convert_time_to_physical_units(torch.tensor([time_in_lattice_units],device='cuda'))
        velocity = (0.01 * (10.195246145610444*torch.sin(2*torch.pi*t) + 1.1298507094850878*torch.sin(4*torch.pi*t) -
                                       0.74105235761620536*torch.sin(6*torch.pi*t) - 0.81494335013637598*torch.sin(8*torch.pi*t) - 0.057746786774581969*torch.sin(10*torch.pi*t) -
                                       0.14077501532395437*torch.sin(12*torch.pi*t) - 0.79031631408142378*torch.sin(14*torch.pi*t) - 0.28476181019884617*torch.sin(16*torch.pi*t) -
                                       0.35987826709561233*torch.sin(18*torch.pi*t) - 0.59760796631747748*torch.sin(20*torch.pi*t) + 0.17632264699428082*torch.sin(22*torch.pi*t) -
                                       0.11470836108705555*torch.sin(24*torch.pi*t) + 0.05591297409225085*torch.sin(26*torch.pi*t) + 0.23368836071036847*torch.sin(28*torch.pi*t) +
                                       0.0077192911097072403*torch.sin(30*torch.pi*t) + 0.079034016576042504*torch.sin(32*torch.pi*t) + 0.1372511216612694*torch.sin(34*torch.pi*t) +
                                       0.02526943589827953*torch.sin(36*torch.pi*t) + 0.053790188842640876*torch.sin(38*torch.pi*t) + 0.10971483024748149*torch.sin(40*torch.pi*t) +
                                       0.02662538788348863*torch.sin(42*torch.pi*t) + 0.023165644549146113*torch.sin(44*torch.pi*t) + 0.091767895244818071*torch.sin(46*torch.pi*t) -
                                       0.085557125711564444*torch.sin(48*torch.pi*t) - 4.0890531286068503*torch.cos(2*torch.pi*t) - 4.6764844416716231*torch.cos(4*torch.pi*t) -
                                       1.7745465713641333*torch.cos(6*torch.pi*t) - 0.58448683056231654*torch.cos(8*torch.pi*t) - 0.17341521542500002*torch.cos(10*torch.pi*t) -
                                       0.71901379564716983*torch.cos(12*torch.pi*t) - 0.31334317510925319*torch.cos(14*torch.pi*t) + 0.21847361832000456*torch.cos(16*torch.pi*t) -
                                       0.25396003397508332*torch.cos(18*torch.pi*t) + 0.46308743289218418*torch.cos(20*torch.pi*t) + 0.37918140506728976*torch.cos(22*torch.pi*t) +
                                       0.059933911166476166*torch.cos(24*torch.pi*t) + 0.36132472621651529*torch.cos(26*torch.pi*t) + 0.040215669127426419*torch.cos(28*torch.pi*t) -
                                       0.0018115535281448131*torch.cos(30*torch.pi*t) + 0.11158280721883845*torch.cos(32*torch.pi*t) - 0.014166264208691706*torch.cos(34*torch.pi*t) -
                                       0.0360887282925745*torch.cos(36*torch.pi*t) + 0.038414238570343262*torch.cos(38*torch.pi*t) - 0.033489241856483984*torch.cos(40*torch.pi*t) -
                                       0.081175286384966017*torch.cos(42*torch.pi*t) - 0.0030576966620926484*torch.cos(44*torch.pi*t) - 0.098321151242134197*torch.cos(46*torch.pi*t)
                                       - 0.11447797868211534*torch.cos(48*torch.pi*t) + 28.072428258316549)).cuda()"""

        # Sine Inflow
        time_in_physical_units = self.unit_converter.convert_time_to_physical_units(self.time_parameter)
        inflow_velocity = torch.tensor([0.01])
        # velocity = inflow_velocity - 0.1 * inflow_velocity * torch.sin(2 * torch.pi * time_in_physical_units / 1)
        velocity = 0.01 * (
            10.195246145610444 * torch.sin(2 * torch.pi * time_in_physical_units)
            + 1.1298507094850878 * torch.sin(4 * torch.pi * time_in_physical_units)
            - 0.74105235761620536 * torch.sin(6 * torch.pi * time_in_physical_units)
            - 0.81494335013637598 * torch.sin(8 * torch.pi * time_in_physical_units)
            - 0.057746786774581969 * torch.sin(10 * torch.pi * time_in_physical_units)
            - 0.14077501532395437 * torch.sin(12 * torch.pi * time_in_physical_units)
            - 0.79031631408142378 * torch.sin(14 * torch.pi * time_in_physical_units)
            - 0.28476181019884617 * torch.sin(16 * torch.pi * time_in_physical_units)
            - 0.35987826709561233 * torch.sin(18 * torch.pi * time_in_physical_units)
            - 0.59760796631747748 * torch.sin(20 * torch.pi * time_in_physical_units)
            + 0.17632264699428082 * torch.sin(22 * torch.pi * time_in_physical_units)
            - 0.11470836108705555 * torch.sin(24 * torch.pi * time_in_physical_units)
            + 0.05591297409225085 * torch.sin(26 * torch.pi * time_in_physical_units)
            + 0.23368836071036847 * torch.sin(28 * torch.pi * time_in_physical_units)
            + 0.0077192911097072403 * torch.sin(30 * torch.pi * time_in_physical_units)
            + 0.079034016576042504 * torch.sin(32 * torch.pi * time_in_physical_units)
            + 0.1372511216612694 * torch.sin(34 * torch.pi * time_in_physical_units)
            + 0.02526943589827953 * torch.sin(36 * torch.pi * time_in_physical_units)
            + 0.053790188842640876 * torch.sin(38 * torch.pi * time_in_physical_units)
            + 0.10971483024748149 * torch.sin(40 * torch.pi * time_in_physical_units)
            + 0.02662538788348863 * torch.sin(42 * torch.pi * time_in_physical_units)
            + 0.023165644549146113 * torch.sin(44 * torch.pi * time_in_physical_units)
            + 0.091767895244818071 * torch.sin(46 * torch.pi * time_in_physical_units)
            - 0.085557125711564444 * torch.sin(48 * torch.pi * time_in_physical_units)
            - 4.0890531286068503 * torch.cos(2 * torch.pi * time_in_physical_units)
            - 4.6764844416716231 * torch.cos(4 * torch.pi * time_in_physical_units)
            - 1.7745465713641333 * torch.cos(6 * torch.pi * time_in_physical_units)
            - 0.58448683056231654 * torch.cos(8 * torch.pi * time_in_physical_units)
            - 0.17341521542500002 * torch.cos(10 * torch.pi * time_in_physical_units)
            - 0.71901379564716983 * torch.cos(12 * torch.pi * time_in_physical_units)
            - 0.31334317510925319 * torch.cos(14 * torch.pi * time_in_physical_units)
            + 0.21847361832000456 * torch.cos(16 * torch.pi * time_in_physical_units)
            - 0.25396003397508332 * torch.cos(18 * torch.pi * time_in_physical_units)
            + 0.46308743289218418 * torch.cos(20 * torch.pi * time_in_physical_units)
            + 0.37918140506728976 * torch.cos(22 * torch.pi * time_in_physical_units)
            + 0.059933911166476166 * torch.cos(24 * torch.pi * time_in_physical_units)
            + 0.36132472621651529 * torch.cos(26 * torch.pi * time_in_physical_units)
            + 0.040215669127426419 * torch.cos(28 * torch.pi * time_in_physical_units)
            - 0.0018115535281448131 * torch.cos(30 * torch.pi * time_in_physical_units)
            + 0.11158280721883845 * torch.cos(32 * torch.pi * time_in_physical_units)
            - 0.014166264208691706 * torch.cos(34 * torch.pi * time_in_physical_units)
            - 0.0360887282925745 * torch.cos(36 * torch.pi * time_in_physical_units)
            + 0.038414238570343262 * torch.cos(38 * torch.pi * time_in_physical_units)
            - 0.033489241856483984 * torch.cos(40 * torch.pi * time_in_physical_units)
            - 0.081175286384966017 * torch.cos(42 * torch.pi * time_in_physical_units)
            - 0.0030576966620926484 * torch.cos(44 * torch.pi * time_in_physical_units)
            - 0.098321151242134197 * torch.cos(46 * torch.pi * time_in_physical_units)
            - 0.11447797868211534 * torch.cos(48 * torch.pi * time_in_physical_units)
            + 28.072428258316549
        )

        # velocity = 0.1 * (
        #     0.47028017170427949 * torch.sin(2 * torch.pi * time_in_physical_units)
        #     + 0.049291723561855294 * torch.sin(4 * torch.pi * time_in_physical_units)
        #     - 0.036244172243928306 * torch.sin(6 * torch.pi * time_in_physical_units)
        #     - 0.037915994177868909 * torch.sin(8 * torch.pi * time_in_physical_units)
        #     - 0.0020258393985765192 * torch.sin(10 * torch.pi * time_in_physical_units)
        #     - 0.0062093431428632126 * torch.sin(12 * torch.pi * time_in_physical_units)
        #     - 0.036171894648237328 * torch.sin(14 * torch.pi * time_in_physical_units)
        #     - 0.01257219843573476 * torch.sin(16 * torch.pi * time_in_physical_units)
        #     - 0.016237640029026883 * torch.sin(18 * torch.pi * time_in_physical_units)
        #     - 0.027121065327231464 * torch.sin(20 * torch.pi * time_in_physical_units)
        #     + 0.0083928339698547873 * torch.sin(22 * torch.pi * time_in_physical_units)
        #     - 0.0051910505058529923 * torch.sin(24 * torch.pi * time_in_physical_units)
        #     + 0.0025699223156927394 * torch.sin(26 * torch.pi * time_in_physical_units)
        #     + 0.010697937787909943 * torch.sin(28 * torch.pi * time_in_physical_units)
        #     + 0.00023296260161105597 * torch.sin(30 * torch.pi * time_in_physical_units)
        #     + 0.0035469075514001883 * torch.sin(32 * torch.pi * time_in_physical_units)
        #     + 0.0062176615169810207 * torch.sin(34 * torch.pi * time_in_physical_units)
        #     + 0.0010510072787308052 * torch.sin(36 * torch.pi * time_in_physical_units)
        #     + 0.0023927216998217643 * torch.sin(38 * torch.pi * time_in_physical_units)
        #     + 0.0049469215011127545 * torch.sin(40 * torch.pi * time_in_physical_units)
        #     + 0.0011415846653079445 * torch.sin(42 * torch.pi * time_in_physical_units)
        #     + 0.0010269565785079524 * torch.sin(44 * torch.pi * time_in_physical_units)
        #     + 0.0041773039991289621 * torch.sin(46 * torch.pi * time_in_physical_units)
        #     - 0.0039025475855080467 * torch.sin(48 * torch.pi * time_in_physical_units)
        #     - 0.19132111724064102 * torch.cos(2 * torch.pi * time_in_physical_units)
        #     - 0.2164383955603976 * torch.cos(4 * torch.pi * time_in_physical_units)
        #     - 0.080722344161411844 * torch.cos(6 * torch.pi * time_in_physical_units)
        #     - 0.02505863830382974 * torch.cos(8 * torch.pi * time_in_physical_units)
        #     - 0.0071277207907017824 * torch.cos(10 * torch.pi * time_in_physical_units)
        #     - 0.032882278793956478 * torch.cos(12 * torch.pi * time_in_physical_units)
        #     - 0.013957206529770063 * torch.cos(14 * torch.pi * time_in_physical_units)
        #     + 0.010247055039322821 * torch.cos(16 * torch.pi * time_in_physical_units)
        #     - 0.011695423220892013 * torch.cos(18 * torch.pi * time_in_physical_units)
        #     + 0.021256910913860401 * torch.cos(20 * torch.pi * time_in_physical_units)
        #     + 0.017185829871684512 * torch.cos(22 * torch.pi * time_in_physical_units)
        #     + 0.0024825077137337055 * torch.cos(24 * torch.pi * time_in_physical_units)
        #     + 0.016408845224302068 * torch.cos(26 * torch.pi * time_in_physical_units)
        #     + 0.0016546947486029548 * torch.cos(28 * torch.pi * time_in_physical_units)
        #     - 0.00020211049875023038 * torch.cos(30 * torch.pi * time_in_physical_units)
        #     + 0.0050761581940564375 * torch.cos(32 * torch.pi * time_in_physical_units)
        #     - 0.00071191950401289597 * torch.cos(34 * torch.pi * time_in_physical_units)
        #     - 0.001671680435802631 * torch.cos(36 * torch.pi * time_in_physical_units)
        #     + 0.0017741578854594308 * torch.cos(38 * torch.pi * time_in_physical_units)
        #     - 0.0015249239139772301 * torch.cos(40 * torch.pi * time_in_physical_units)
        #     - 0.003661588994030724 * torch.cos(42 * torch.pi * time_in_physical_units)
        #     - 7.2029745258647002e-5 * torch.cos(44 * torch.pi * time_in_physical_units)
        #     - 0.0044322912217235515 * torch.cos(46 * torch.pi * time_in_physical_units)
        #     - 0.0051691966179672451 * torch.cos(48 * torch.pi * time_in_physical_units)
        #     + 1.2676234781746032
        # )

        pre_factor = time_in_physical_units**2 / (time_in_physical_units**2 + 0.1)
        velocity = pre_factor * velocity
        velocity_lattice_units = self.unit_converter.convert_velocity_to_lattice_units(velocity)

        y_0_upper = 0.00574
        z_0_upper = 0.0116
        radius_upper = 0.00102

        y_0_lower = 0.00483
        z_0_lower = 0.00184
        radius_lower = 0.0017

        Y = self.west_meshgrid_const[1]
        Z = self.west_meshgrid_const[2]

        # velocity_scaling_factor = torch.where(
        #     torch.sqrt((Y-y_0_lower)*(Y-y_0_lower)+(Z-z_0_lower)*(Z-z_0_lower)) < radius_lower,
        #     1.0 - torch.pow(torch.sqrt((Y-y_0_lower)*(Y-y_0_lower)+(Z-z_0_lower)*(Z-z_0_lower)) / radius_lower, 2),
        #     torch.where(
        #         torch.sqrt((Y-y_0_upper)*(Y-y_0_upper)+(Z-z_0_upper)*(Z-z_0_upper)) < radius_upper,
        #         1.0 - torch.pow(torch.sqrt((Y-y_0_upper)*(Y-y_0_upper)+(Z-z_0_upper)*(Z-z_0_upper)) / radius_upper, 2),
        #         0.0
        #     )
        # )

        y_0_single = 0.00318  # 0.009823829787 - 0.00318
        z_0_single = 0.00745
        radius_single = 0.00185
        velocity_scaling_factor = torch.where(
            torch.sqrt((Y - y_0_single) * (Y - y_0_single) + (Z - z_0_single) * (Z - z_0_single)) < radius_single,
            1.0 - torch.pow(torch.sqrt((Y - y_0_single) * (Y - y_0_single) + (Z - z_0_single) * (Z - z_0_single)) / radius_single, 2),
            0.0,
        )

        cell_size = 2 * 0.001649 / (60 - 13)
        offset = cell_size * 12
        radius = cell_size * 47 * 0.5

        # velocity_scaling_factor = 1.0 - torch.pow((self.west_meshgrid_const[1] - radius - offset) / radius, 2)  ####1-((x-0.0575-0.035)/0.06)^2

        # print(velocity_scaling_factor.shape)
        velocity_x = 1.5 * velocity_scaling_factor * velocity_lattice_units
        velocity_y = torch.zeros_like(velocity_x)
        velocity_z = torch.zeros_like(velocity_x)
        velocity_x = velocity_x.unsqueeze(0)
        velocity_y = velocity_y.unsqueeze(0)
        velocity_z = velocity_z.unsqueeze(0)
        return torch.cat([velocity_x, velocity_y, velocity_z], dim=0)

    def get_north_wall_velocity(self) -> torch.Tensor:
        return torch.cat(
            [torch.zeros_like(self.north_meshgrid_const[0]), torch.zeros_like(self.north_meshgrid_const[0]), torch.zeros_like(self.north_meshgrid_const[0])],
            dim=0,
        )

    def get_south_wall_velocity(self) -> torch.Tensor:
        return torch.cat(
            [torch.zeros_like(self.south_meshgrid_const[0]), torch.zeros_like(self.south_meshgrid_const[0]), torch.zeros_like(self.south_meshgrid_const[0])],
            dim=0,
        )

    def get_top_wall_velocity(self) -> torch.Tensor:
        return torch.cat(
            [torch.zeros_like(self.top_meshgrid_const[0]), torch.zeros_like(self.top_meshgrid_const[0]), torch.zeros_like(self.top_meshgrid_const[0])], dim=0
        )

    def get_bottom_wall_velocity(self) -> torch.Tensor:
        return torch.cat(
            [torch.zeros_like(self.bottom_meshgrid_const[0]), torch.zeros_like(self.bottom_meshgrid_const[0]), torch.zeros_like(self.bottom_meshgrid_const[0])],
            dim=0,
        )

    def forward(self, location: BoundaryLocation) -> torch.Tensor:
        self.time_parameter = self.time_parameter + 1
        if location == BoundaryLocation.East:
            return self.get_east_wall_velocity()
        elif location == BoundaryLocation.West:
            return self.get_west_wall_velocity()
        elif location == BoundaryLocation.North:
            return self.get_north_wall_velocity()
        elif location == BoundaryLocation.South:
            return self.get_south_wall_velocity()
        elif location == BoundaryLocation.Top:
            return self.get_top_wall_velocity()
        else:
            return self.get_bottom_wall_velocity()

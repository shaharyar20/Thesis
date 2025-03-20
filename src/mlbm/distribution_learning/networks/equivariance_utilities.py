import torch
from typing import List


def mirror(x: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [x[:, 0, None], x[:, 1, None], x[:, 4, None], x[:, 3, None], x[:, 2, None], x[:, 8, None], x[:, 7, None], x[:, 6, None], x[:, 5, None]], dim=-1
    )


def rotate(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cat([x[:, 0, None], x[:, 1:5].roll(k, 1), x[:, 5:].roll(k, 1)], dim=-1)


def rotate_data(x: torch.Tensor) -> List[torch.Tensor]:
    return [
        x,
        rotate(x, 1),
        rotate(x, 2),
        rotate(x, 3),
        mirror(x),
        rotate(mirror(x), 1),
        rotate(mirror(x), 2),
        rotate(mirror(x), 3),
    ]


def derotate_data(x: List[torch.Tensor]) -> List[torch.Tensor]:
    return [
        x[0],
        rotate(x[1], -1),
        rotate(x[2], -2),
        rotate(x[3], -3),
        mirror(x[4]),
        mirror(rotate(x[5], -1)),
        mirror(rotate(x[6], -2)),
        mirror(rotate(x[7], -3)),
    ]

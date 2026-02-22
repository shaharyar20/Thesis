import torch


def decompose_s_d2q9_nat(f_i: torch.Tensor, c: torch.Tensor, includeT: bool = True, includeQ: bool = True) -> torch.Tensor:
    c = c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    M20 = torch.sum(c[0, :] ** 2 * f_i, dim=0)
    M02 = torch.sum(c[1, :] ** 2 * f_i, dim=0)
    M11 = torch.sum(c[0, :] * c[1, :] * f_i, dim=0)

    M21 = torch.sum(c[0, :] ** 2 * c[1, :] * f_i, dim=0)
    M12 = torch.sum(c[0, :] * c[1, :] ** 2 * f_i, dim=0)

    T_nat = M20 + M02
    N_nat = M20 - M02
    Pi_nat = M11
    Q_xxy_nat = M21
    Q_xyy_nat = M12

    d_i = torch.cat(
        (
            (torch.zeros_like(N_nat)),
            (N_nat / 4.0),
            (-N_nat / 4.0),
            (N_nat / 4.0),
            (-N_nat / 4.0),
            (Pi_nat / 4.0),
            (-Pi_nat / 4.0),
            (Pi_nat / 4.0),
            (-Pi_nat / 4.0),
        ),
        dim=0,
    ).view(9, f_i.shape[1], f_i.shape[2], f_i.shape[3])

    if includeT:
        t_i = torch.cat(
            (
                -T_nat,
                (T_nat / 4.0),
                (T_nat / 4.0),
                (T_nat / 4.0),
                (T_nat / 4.0),
                torch.zeros_like(T_nat),
                torch.zeros_like(T_nat),
                torch.zeros_like(T_nat),
                torch.zeros_like(T_nat),
            ),
            dim=0,
        ).view(9, f_i.shape[1], f_i.shape[2], f_i.shape[3])
    else:
        t_i = torch.zeros_like(d_i)

    if includeQ:
        q_i = torch.cat(
            (
                torch.zeros_like(T_nat),
                (-Q_xyy_nat / 2.0),
                (-Q_xxy_nat / 2.0),
                (Q_xyy_nat / 2.0),
                (Q_xxy_nat / 2.0),
                ((Q_xyy_nat + Q_xxy_nat) / 4.0),
                ((-Q_xyy_nat + Q_xxy_nat) / 4.0),
                ((-Q_xyy_nat - Q_xxy_nat) / 4.0),
                ((Q_xyy_nat - Q_xxy_nat) / 4.0),
            ),
            dim=0,
        ).view(9, f_i.shape[1], f_i.shape[2], f_i.shape[3])
    else:
        q_i = torch.zeros_like(d_i)

    return d_i + t_i + q_i


def decompose_s_d3q27_nat(f_i: torch.Tensor, c: torch.Tensor, includeT: bool = False, includeQ: bool = False) -> torch.Tensor:
    c = c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    M200 = torch.sum(c[0, :] ** 2 * f_i, dim=0)
    M020 = torch.sum(c[1, :] ** 2 * f_i, dim=0)
    M002 = torch.sum(c[2, :] ** 2 * f_i, dim=0)
    M110 = torch.sum(c[0, :] * c[1, :] * f_i, dim=0)
    M101 = torch.sum(c[0, :] * c[2, :] * f_i, dim=0)
    M011 = torch.sum(c[1, :] * c[2, :] * f_i, dim=0)

    N_yz_nat = M020 - M002
    N_xz_nat = M200 - M002

    Pi_xy_nat = M110
    Pi_yz_nat = M011
    Pi_xz_nat = M101

    d_i = torch.stack(
        [
            torch.zeros_like(N_xz_nat),  # 0, ( 0, 0, 0)
            ((2.0 * N_xz_nat - N_yz_nat) / 6.0),  # 1, ( 1, 0, 0)
            ((2.0 * N_xz_nat - N_yz_nat) / 6.0),  # 2, (-1, 0, 0)
            ((-N_xz_nat + 2.0 * N_yz_nat) / 6.0),  # 3, ( 0, 1, 0)
            ((-N_xz_nat + 2.0 * N_yz_nat) / 6.0),  # 4, ( 0,-1, 0)
            ((-N_xz_nat - N_yz_nat) / 6.0),  # 5, ( 0, 0, 1)
            ((-N_xz_nat - N_yz_nat) / 6.0),  # 6, ( 0, 0,-1)
            (Pi_xy_nat / 4.0),  # 7, ( 1, 1, 0)
            (-Pi_xy_nat / 4.0),  # 8, ( 1,-1, 0)
            (-Pi_xy_nat / 4.0),  # 9, (-1, 1, 0)
            (Pi_xy_nat / 4.0),  # 10,(-1,-1, 0)
            (Pi_xz_nat / 4.0),  # 11,( 1, 0, 1)
            (-Pi_xz_nat / 4.0),  # 12,( 1, 0,-1)
            (-Pi_xz_nat / 4.0),  # 13,(-1, 0, 1)
            (Pi_xz_nat / 4.0),  # 14,(-1, 0,-1)
            (Pi_yz_nat / 4.0),  # 15,( 0, 1, 1)
            (-Pi_yz_nat / 4.0),  # 16,( 0, 1,-1)
            (-Pi_yz_nat / 4.0),  # 17,( 0,-1, 1)
            (Pi_yz_nat / 4.0),  # 18,( 0,-1,-1)
            torch.zeros_like(N_xz_nat),  # 19,( 1, 1, 1)
            torch.zeros_like(N_xz_nat),  # 20,( 1, 1,-1)
            torch.zeros_like(N_xz_nat),  # 21,( 1,-1, 1)
            torch.zeros_like(N_xz_nat),  # 22,( 1,-1,-1)
            torch.zeros_like(N_xz_nat),  # 23,(-1, 1, 1)
            torch.zeros_like(N_xz_nat),  # 24,(-1, 1,-1)
            torch.zeros_like(N_xz_nat),  # 25,(-1,-1, 1)
            torch.zeros_like(N_xz_nat),  # 26,(-1,-1,-1)
        ]
    )

    if includeT:
        T_nat = M200 + M020 + M002
        t_i = torch.stack(
            [
                (-T_nat),  # 0, ( 0, 0, 0)
                (T_nat / 6.0),  # 1, ( 1, 0, 0)
                (T_nat / 6.0),  # 2, (-1, 0, 0)
                (T_nat / 6.0),  # 3, ( 0, 1, 0)
                (T_nat / 6.0),  # 4, ( 0,-1, 0)
                (T_nat / 6.0),  # 5, ( 0, 0, 1)
                (T_nat / 6.0),  # 6, ( 0, 0,-1)
                torch.zeros_like(T_nat),  # 7, ( 1, 1, 0)
                torch.zeros_like(T_nat),  # 8, ( 1,-1, 0)
                torch.zeros_like(T_nat),  # 9, (-1, 1, 0)
                torch.zeros_like(T_nat),  # 10,(-1,-1, 0)
                torch.zeros_like(T_nat),  # 11,( 1, 0, 1)
                torch.zeros_like(T_nat),  # 12,( 1, 0,-1)
                torch.zeros_like(T_nat),  # 13,(-1, 0, 1)
                torch.zeros_like(T_nat),  # 14,(-1, 0,-1)
                torch.zeros_like(T_nat),  # 15,( 0, 1, 1)
                torch.zeros_like(T_nat),  # 16,( 0, 1,-1)
                torch.zeros_like(T_nat),  # 17,( 0,-1, 1)
                torch.zeros_like(T_nat),  # 18,( 0,-1,-1)
                torch.zeros_like(T_nat),  # 19,( 1, 1, 1)
                torch.zeros_like(T_nat),  # 20,( 1, 1,-1)
                torch.zeros_like(T_nat),  # 21,( 1,-1, 1)
                torch.zeros_like(T_nat),  # 22,( 1,-1,-1)
                torch.zeros_like(T_nat),  # 23,(-1, 1, 1)
                torch.zeros_like(T_nat),  # 24,(-1, 1,-1)
                torch.zeros_like(T_nat),  # 25,(-1,-1, 1)
                torch.zeros_like(T_nat),  # 26,(-1,-1,-1)
            ]
        )
        d_i = d_i + t_i

    if includeQ:
        M111 = torch.sum(c[0, :] * c[1, :] * c[2, :] * f_i, dim=0)
        M210 = torch.sum(c[0, :] ** 2 * c[1, :] * f_i, dim=0)
        M201 = torch.sum(c[0, :] ** 2 * c[2, :] * f_i, dim=0)
        M021 = torch.sum(c[1, :] ** 2 * c[2, :] * f_i, dim=0)
        M120 = torch.sum(c[0, :] * c[1, :] ** 2 * f_i, dim=0)
        M102 = torch.sum(c[0, :] * c[2, :] ** 2 * f_i, dim=0)
        M012 = torch.sum(c[1, :] * c[2, :] ** 2 * f_i, dim=0)
        Q_xxy_nat = M210
        Q_xyy_nat = M120
        Q_xzz_nat = M102
        Q_yzz_nat = M012
        Q_yyz_nat = M021
        Q_xxz_nat = M201
        Q_xyz_nat = M111
        q_i = torch.stack(
            [
                torch.zeros_like(M111),  # 0, ( 0, 0, 0)
                (-(Q_xyy_nat + Q_xzz_nat) / 2.0),  # 1, ( 1, 0, 0)
                ((Q_xyy_nat + Q_xzz_nat) / 2.0),  # 2, (-1, 0, 0)
                (-(Q_xxy_nat + Q_yzz_nat) / 2.0),  # 3, ( 0, 1, 0)
                ((Q_xxy_nat + Q_yzz_nat) / 2.0),  # 4, ( 0,-1, 0)
                (-(Q_xxz_nat + Q_yyz_nat) / 2.0),  # 5, ( 0, 0, 1)
                ((Q_xxz_nat + Q_yyz_nat) / 2.0),  # 6, ( 0, 0,-1)
                ((Q_xxy_nat + Q_xyy_nat) / 4.0),  # 7, ( 1, 1, 0)
                ((-Q_xxy_nat + Q_xyy_nat) / 4.0),  # 8, ( 1,-1, 0)
                ((Q_xxy_nat - Q_xyy_nat) / 4.0),  # 9, (-1, 1, 0)
                ((-Q_xxy_nat - Q_xyy_nat) / 4.0),  # 10,(-1,-1, 0)
                ((Q_xxz_nat + Q_xzz_nat) / 4.0),  # 11,( 1, 0, 1)
                ((-Q_xxz_nat + Q_xzz_nat) / 4.0),  # 12,( 1, 0,-1)
                ((Q_xxz_nat - Q_xzz_nat) / 4.0),  # 13,(-1, 0, 1)
                ((-Q_xxz_nat - Q_xzz_nat) / 4.0),  # 14,(-1, 0,-1)
                ((Q_yyz_nat + Q_yzz_nat) / 4.0),  # 15,( 0, 1, 1)
                ((-Q_yyz_nat + Q_yzz_nat) / 4.0),  # 16,( 0, 1,-1)
                ((Q_yyz_nat - Q_yzz_nat) / 4.0),  # 17,( 0,-1, 1)
                ((-Q_yyz_nat - Q_yzz_nat) / 4.0),  # 18,( 0,-1,-1)
                (Q_xyz_nat / 8.0),  # 19,( 1, 1, 1)
                (-Q_xyz_nat / 8.0),  # 20,( 1, 1,-1)
                (-Q_xyz_nat / 8.0),  # 21,( 1,-1, 1)
                (Q_xyz_nat / 8.0),  # 22,( 1,-1,-1)
                (-Q_xyz_nat / 8.0),  # 23,(-1, 1, 1)
                (Q_xyz_nat / 8.0),  # 24,(-1, 1,-1)
                (Q_xyz_nat / 8.0),  # 25,(-1,-1, 1)
                (-Q_xyz_nat / 8.0),  # 26,(-1,-1,-1)
            ]
        )
        d_i = d_i + q_i

    return d_i


def decompose_s_d3q19_nat(f_i: torch.Tensor, c: torch.Tensor, includeT: bool = False, includeQ: bool = False) -> torch.Tensor:
    c = c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    M200 = torch.sum(c[0, :] ** 2 * f_i, dim=0)
    M020 = torch.sum(c[1, :] ** 2 * f_i, dim=0)
    M002 = torch.sum(c[2, :] ** 2 * f_i, dim=0)
    M110 = torch.sum(c[0, :] * c[1, :] * f_i, dim=0)
    M101 = torch.sum(c[0, :] * c[2, :] * f_i, dim=0)
    M011 = torch.sum(c[1, :] * c[2, :] * f_i, dim=0)

    M210 = torch.sum(c[0, :] ** 2 * c[1, :] * f_i, dim=0)
    M201 = torch.sum(c[0, :] ** 2 * c[2, :] * f_i, dim=0)
    M021 = torch.sum(c[1, :] ** 2 * c[2, :] * f_i, dim=0)
    M120 = torch.sum(c[0, :] * c[1, :] ** 2 * f_i, dim=0)
    M102 = torch.sum(c[0, :] * c[2, :] ** 2 * f_i, dim=0)
    M012 = torch.sum(c[1, :] * c[2, :] ** 2 * f_i, dim=0)

    T_nat = M200 + M020 + M002

    N_yz_nat = M020 - M002
    N_xz_nat = M200 - M002

    Pi_xy_nat = M110
    Pi_yz_nat = M011
    Pi_xz_nat = M101

    Q_xxy_nat = M210
    Q_xyy_nat = M120
    Q_xzz_nat = M102
    Q_yzz_nat = M012
    Q_yyz_nat = M021
    Q_xxz_nat = M201

    d_i = torch.cat(
        (
            torch.zeros_like(T_nat),  # 0, ( 0, 0, 0)
            ((2.0 * N_xz_nat - N_yz_nat) / 6.0),  # 1, ( 1, 0, 0)
            ((2.0 * N_xz_nat - N_yz_nat) / 6.0),  # 2, (-1, 0, 0)
            ((-N_xz_nat + 2.0 * N_yz_nat) / 6.0),  # 3, ( 0, 1, 0)
            ((-N_xz_nat + 2.0 * N_yz_nat) / 6.0),  # 4, ( 0,-1, 0)
            ((-N_xz_nat - N_yz_nat) / 6.0),  # 5, ( 0, 0, 1)
            ((-N_xz_nat - N_yz_nat) / 6.0),  # 6, ( 0, 0,-1)
            (Pi_xy_nat / 4.0),  # 7, ( 1, 1, 0)
            (-Pi_xy_nat / 4.0),  # 8, ( 1,-1, 0)
            (-Pi_xy_nat / 4.0),  # 9, (-1, 1, 0)
            (Pi_xy_nat / 4.0),  # 10,(-1,-1, 0)
            (Pi_xz_nat / 4.0),  # 11,( 1, 0, 1)
            (-Pi_xz_nat / 4.0),  # 12,( 1, 0,-1)
            (-Pi_xz_nat / 4.0),  # 13,(-1, 0, 1)
            (Pi_xz_nat / 4.0),  # 14,(-1, 0,-1)
            (Pi_yz_nat / 4.0),  # 15,( 0, 1, 1)
            (-Pi_yz_nat / 4.0),  # 16,( 0, 1,-1)
            (-Pi_yz_nat / 4.0),  # 17,( 0,-1, 1)
            (Pi_yz_nat / 4.0),
        ),  # 18,( 0,-1,-1)
        dim=0,
    ).view(19, f_i.shape[1], f_i.shape[2], f_i.shape[3])

    if includeT:
        t_i = torch.cat(
            (
                (-T_nat),  # 0, ( 0, 0, 0)
                (T_nat / 6.0),  # 1, ( 1, 0, 0)
                (T_nat / 6.0),  # 2, (-1, 0, 0)
                (T_nat / 6.0),  # 3, ( 0, 1, 0)
                (T_nat / 6.0),  # 4, ( 0,-1, 0)
                (T_nat / 6.0),  # 5, ( 0, 0, 1)
                (T_nat / 6.0),  # 6, ( 0, 0,-1)
                torch.zeros_like(T_nat),  # 7, ( 1, 1, 0)
                torch.zeros_like(T_nat),  # 8, ( 1,-1, 0)
                torch.zeros_like(T_nat),  # 9, (-1, 1, 0)
                torch.zeros_like(T_nat),  # 10,(-1,-1, 0)
                torch.zeros_like(T_nat),  # 11,( 1, 0, 1)
                torch.zeros_like(T_nat),  # 12,( 1, 0,-1)
                torch.zeros_like(T_nat),  # 13,(-1, 0, 1)
                torch.zeros_like(T_nat),  # 14,(-1, 0,-1)
                torch.zeros_like(T_nat),  # 15,( 0, 1, 1)
                torch.zeros_like(T_nat),  # 16,( 0, 1,-1)
                torch.zeros_like(T_nat),  # 17,( 0,-1, 1)
                torch.zeros_like(T_nat),
            ),  # 18,( 0,-1,-1)
            dim=0,
        ).view(19, f_i.shape[1], f_i.shape[2], f_i.shape[3])
    else:
        t_i = torch.zeros_like(d_i)

    if includeQ:
        q_i = torch.cat(
            (
                torch.zeros_like(T_nat),  # 0, ( 0, 0, 0)
                (-(Q_xyy_nat + Q_xzz_nat) / 2.0),  # 1, ( 1, 0, 0)
                ((Q_xyy_nat + Q_xzz_nat) / 2.0),  # 2, (-1, 0, 0)
                (-(Q_xxy_nat + Q_yzz_nat) / 2.0),  # 3, ( 0, 1, 0)
                ((Q_xxy_nat + Q_yzz_nat) / 2.0),  # 4, ( 0,-1, 0)
                (-(Q_xxz_nat + Q_yyz_nat) / 2.0),  # 5, ( 0, 0, 1)
                ((Q_xxz_nat + Q_yyz_nat) / 2.0),  # 6, ( 0, 0,-1)
                ((Q_xxy_nat + Q_xyy_nat) / 4.0),  # 7, ( 1, 1, 0)
                ((-Q_xxy_nat + Q_xyy_nat) / 4.0),  # 8, ( 1,-1, 0)
                ((Q_xxy_nat - Q_xyy_nat) / 4.0),  # 9, (-1, 1, 0)
                ((-Q_xxy_nat - Q_xyy_nat) / 4.0),  # 10,(-1,-1, 0)
                ((Q_xxz_nat + Q_xzz_nat) / 4.0),  # 11,( 1, 0, 1)
                ((-Q_xxz_nat + Q_xzz_nat) / 4.0),  # 12,( 1, 0,-1)
                ((Q_xxz_nat - Q_xzz_nat) / 4.0),  # 13,(-1, 0, 1)
                ((-Q_xxz_nat - Q_xzz_nat) / 4.0),  # 14,(-1, 0,-1)
                ((Q_yyz_nat + Q_yzz_nat) / 4.0),  # 15,( 0, 1, 1)
                ((-Q_yyz_nat + Q_yzz_nat) / 4.0),  # 16,( 0, 1,-1)
                ((Q_yyz_nat - Q_yzz_nat) / 4.0),  # 17,( 0,-1, 1)
                ((-Q_yyz_nat - Q_yzz_nat) / 4.0),
            ),  # 18,( 0,-1,-1)
            dim=0,
        ).view(19, f_i.shape[1], f_i.shape[2], f_i.shape[3])
    else:
        q_i = torch.zeros_like(d_i)

    return d_i + t_i + q_i

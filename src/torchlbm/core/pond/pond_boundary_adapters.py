"""PonD boundary modules: domain-edge inflow/outflow and immersed solid walls.

Edges: PondWallInletBoundaryUpdate (fixed inflow), PondZeroGradientBoundaryUpdate
(open), PondJetInletBoundaryUpdate (jet patch). Solid bodies: PondNoSlipWallUpdate
(bounce-back), PondMovingWall (moving bounce-back), and PondSdfNoSlipWall -- a sub-cell
no-slip immersed wall driven by a signed-distance field (staircase-free). Each rewrites
the populations/moments in its region after advection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchlbm.compressible_node_data import CompressibleNodeData
from torchlbm.core.pond.pond_equilibrium import PondEquilibrium

_SIDE_AXIS = {"west": (1, 0), "east": (1, 1), "south": (2, 0), "north": (2, 1)}


class PondWallInletBoundaryUpdate(nn.Module):
    """Fixed (rho, u, T) inflow along a domain side (e.g. the supersonic west inlet)."""

    def __init__(self, side, density, velocity, temperature, num_halo_cells, equilibrium: PondEquilibrium):
        super(PondWallInletBoundaryUpdate, self).__init__()
        self.side = side
        self.num_halo_cells = num_halo_cells
        self.equilibrium = equilibrium
        self.register_buffer("density", torch.tensor(float(density)))
        self.register_buffer("velocity", torch.tensor(velocity, dtype=torch.get_default_dtype()))
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def _boundary_slice(self, axis, low):
        sl = [slice(None), slice(None), slice(None), slice(None)]
        sl[axis] = slice(0, self.num_halo_cells + 1) if low else slice(-(self.num_halo_cells + 1), None)
        return tuple(sl)

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        axis, low = _SIDE_AXIS[self.side]
        msl = self._boundary_slice(axis, low)
        sc = msl[1:]

        node_data.moments.density[sc] = self.density
        for d in range(3):
            node_data.moments.velocity[(d,) + sc] = self.velocity[d]
        node_data.moments.temperature[sc] = self.temperature
        node_data.moments.energy[sc] = (
            self.equilibrium.cv * self.temperature + 0.5 * torch.sum(self.velocity ** 2)
        )

        rho = node_data.moments.density
        u = node_data.moments.velocity
        temp = node_data.moments.temperature
        f_eq, g_eq = self.equilibrium.equilibria(rho, temp)
        node_data.distributions.vel_old_population[msl] = f_eq[msl]
        node_data.distributions.temp_old_population[msl] = g_eq[msl]
        return node_data


class PondNoSlipWallUpdate(nn.Module):
    """Bounce-back solid: zero velocity (optionally fixed T) inside the mask."""

    def __init__(self, equilibrium: PondEquilibrium, wall_temperature=None):
        super(PondNoSlipWallUpdate, self).__init__()
        self.equilibrium = equilibrium
        self.wall_temperature = wall_temperature

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        mask = node_data.bounce_back_mask
        if mask is None:
            return node_data
        solid = (mask > 0)
        m = node_data.moments
        u_zero = torch.where(solid.unsqueeze(0), torch.zeros_like(m.velocity), m.velocity)
        m.velocity = u_zero
        if self.wall_temperature is not None:
            m.temperature = torch.where(
                solid, torch.full_like(m.temperature, float(self.wall_temperature)), m.temperature
            )
        m.energy = torch.where(
            solid,
            self.equilibrium.cv * m.temperature + 0.5 * torch.sum(u_zero ** 2, dim=0),
            m.energy,
        )
        f_eq, g_eq = self.equilibrium.equilibria(m.density, m.temperature)
        solid_q = solid.unsqueeze(0)
        node_data.distributions.vel_old_population = torch.where(
            solid_q, f_eq, node_data.distributions.vel_old_population
        )
        node_data.distributions.temp_old_population = torch.where(
            solid_q, g_eq, node_data.distributions.temp_old_population
        )
        return node_data



class PondMovingWall(nn.Module):
    """Bounce-back solid moving at a fixed wall velocity (used by the moving frame)."""

    def __init__(self, equilibrium: PondEquilibrium, wall_velocity, wall_temperature):
        super(PondMovingWall, self).__init__()
        self.equilibrium = equilibrium
        self.wall_temperature = float(wall_temperature)
        wv = torch.zeros(3, dtype=torch.get_default_dtype())
        wv[: len(wall_velocity)] = torch.tensor(wall_velocity, dtype=torch.get_default_dtype())
        self.register_buffer("wall_velocity", wv.reshape(3, 1, 1, 1))

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        mask = node_data.bounce_back_mask
        if mask is None:
            return node_data
        solid = mask > 0
        solid_q = solid.unsqueeze(0)
        m = node_data.moments
        m.velocity = torch.where(solid_q, self.wall_velocity.expand_as(m.velocity), m.velocity)
        m.temperature = torch.where(
            solid, torch.full_like(m.temperature, self.wall_temperature), m.temperature
        )
        ke = 0.5 * float(torch.sum(self.wall_velocity ** 2))
        m.energy = torch.where(
            solid, torch.full_like(m.temperature, self.equilibrium.cv * self.wall_temperature + ke), m.energy
        )
        f_eq, g_eq = self.equilibrium.equilibria(m.density, m.temperature)
        node_data.distributions.vel_old_population = torch.where(
            solid_q, f_eq, node_data.distributions.vel_old_population
        )
        node_data.distributions.temp_old_population = torch.where(
            solid_q, g_eq, node_data.distributions.temp_old_population
        )
        return node_data


class PondJetInletBoundaryUpdate(nn.Module):
    """Circular inflow patch that keeps injecting the jet state on one side."""

    def __init__(self, equilibrium: PondEquilibrium, density, velocity, temperature,
                 jet_radius, num_halo_cells, jet_center=None, side="west"):
        super(PondJetInletBoundaryUpdate, self).__init__()
        if side != "west":
            raise ValueError("PondJetInletBoundaryUpdate is hardcoded for side='west'")
        self.equilibrium = equilibrium
        self.side = side
        self.jet_radius = float(jet_radius)
        self.num_halo_cells = int(num_halo_cells)
        self.jet_center = None if jet_center is None else float(jet_center)
        self.register_buffer("density", torch.tensor(float(density)))
        vel = torch.zeros(3, dtype=torch.get_default_dtype())
        vel[: len(velocity)] = torch.tensor(velocity, dtype=torch.get_default_dtype())
        self.register_buffer("velocity", vel)
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def _region(self, density):
        nx, ny = density.shape[0], density.shape[1]
        dev = density.device
        xi = torch.arange(nx, device=dev).reshape(nx, 1, 1)
        yi = torch.arange(ny, device=dev).reshape(1, ny, 1)
        yc = ny / 2.0 if self.jet_center is None else self.jet_center
        near = xi < (self.num_halo_cells + 1)
        strip = (yi - yc).abs() <= self.jet_radius
        return near & strip

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        m = node_data.moments
        region = self._region(m.density)
        rq = region.unsqueeze(0)

        m.density = torch.where(region, self.density.to(m.density.dtype), m.density)
        m.temperature = torch.where(region, self.temperature.to(m.temperature.dtype), m.temperature)
        vel = self.velocity.reshape(3, 1, 1, 1)
        m.velocity = torch.where(rq, vel.expand_as(m.velocity), m.velocity)
        ke = 0.5 * float(torch.sum(self.velocity ** 2))
        m.energy = torch.where(
            region, torch.full_like(m.energy, self.equilibrium.cv * float(self.temperature) + ke), m.energy
        )

        f_eq, g_eq = self.equilibrium.equilibria(m.density, m.temperature)
        node_data.distributions.vel_old_population = torch.where(
            rq, f_eq, node_data.distributions.vel_old_population
        )
        node_data.distributions.temp_old_population = torch.where(
            rq, g_eq, node_data.distributions.temp_old_population
        )
        return node_data


class PondZeroGradientBoundaryUpdate(nn.Module):
    """Open boundary: copy the interior into the halo (zero gradient)."""

    def __init__(self, side, num_halo_cells):
        super(PondZeroGradientBoundaryUpdate, self).__init__()
        self.side = side
        self.num_halo_cells = num_halo_cells

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        axis, low = _SIDE_AXIS[self.side]
        nh = self.num_halo_cells

        def copy_axis(tensor, tax):
            src = nh if low else tensor.shape[tax] - nh - 1
            index = torch.tensor([src], device=tensor.device)
            ref = torch.index_select(tensor, tax, index)
            halo = slice(0, nh) if low else slice(-nh, None)
            sl = [slice(None)] * tensor.dim()
            sl[tax] = halo
            reps = [1] * tensor.dim()
            reps[tax] = nh
            tensor[tuple(sl)] = ref.repeat(reps)
            return tensor

        d = node_data.distributions
        d.vel_old_population = copy_axis(d.vel_old_population, axis)
        d.temp_old_population = copy_axis(d.temp_old_population, axis)
        m = node_data.moments
        m.density = copy_axis(m.density, axis - 1)
        m.velocity = copy_axis(m.velocity, axis)
        m.temperature = copy_axis(m.temperature, axis - 1)
        m.energy = copy_axis(m.energy, axis - 1)
        return node_data


class PondSdfNoSlipWall(nn.Module):
    """Sub-cell no-slip immersed wall from a signed-distance field (staircase-free).

    Builds a one-cell fluid band just outside the body; each band cell gets a no-slip
    velocity scaled by its sub-cell distance s = d_w/(d_w+1), a wall temperature
    (isothermal if given, else the neighbour value = adiabatic), and a density from
    dp/dn = 0. Solid interior is frozen at the wall state.
    """

    def __init__(self, equilibrium: PondEquilibrium, signed_distance: torch.Tensor,
                 wall_velocity=(0.0, 0.0), wall_temperature=None, dilation=1,
                 num_neighbors=4, idw_power=2.0, density_mode="pressure"):
        super(PondSdfNoSlipWall, self).__init__()
        if density_mode not in ("pressure", "own"):
            raise ValueError(f"unknown density_mode {density_mode!r}")
        self.density_mode = density_mode
        self.cv = equilibrium.cv
        self.dimension = equilibrium.dimension
        self.wall_temperature = None if wall_temperature is None else float(wall_temperature)
        self.dilation = int(dilation)
        self.num_neighbors = int(num_neighbors)
        self.idw_power = float(idw_power)
        self.register_buffer("weights", equilibrium.w.clone())
        wv = torch.zeros(2, dtype=torch.get_default_dtype())
        wv[: len(wall_velocity)] = torch.tensor(wall_velocity, dtype=torch.get_default_dtype())
        self.register_buffer("wall_velocity", wv)
        self.rebuild_geometry(signed_distance)

    def rebuild_geometry(self, signed_distance: torch.Tensor) -> None:
        sdf = signed_distance
        if sdf.dim() == 3:
            sdf = sdf[..., 0]
        nx, ny = sdf.shape
        dev = sdf.device
        solid = sdf < 0.0
        k = 2 * self.dilation + 1
        dil = F.max_pool2d(solid.to(sdf.dtype)[None, None], kernel_size=k, stride=1,
                           padding=self.dilation)[0, 0] > 0
        boundary = dil & (~solid)
        admissible = (~solid) & (~boundary)

        gx, gy = torch.gradient(sdf)
        gmag = torch.sqrt(gx * gx + gy * gy).clamp(min=1e-12)
        nhx, nhy = gx / gmag, gy / gmag

        ii, jj = torch.meshgrid(torch.arange(nx, device=dev, dtype=sdf.dtype),
                                torch.arange(ny, device=dev, dtype=sdf.dtype), indexing="ij")
        b_i, b_j = ii[boundary], jj[boundary]
        self.register_buffer("bnd_flat", (b_i.long() * ny + b_j.long()))
        self.register_buffer("d_w", sdf[boundary].clamp(min=0.0))
        self.register_buffer("solid_flat", torch.where(solid.reshape(-1))[0])

        adm_i, adm_j = ii[admissible], jj[admissible]
        adm_flat = (adm_i.long() * ny + adm_j.long())
        adm_pos = torch.stack([adm_i, adm_j], dim=1)
        ref = torch.stack([b_i + nhx[boundary], b_j + nhy[boundary]], dim=1)
        dist = torch.cdist(ref, adm_pos)
        K = min(self.num_neighbors, adm_pos.shape[0])
        dmin, nn_idx = torch.topk(dist, K, dim=1, largest=False)
        w = dmin.clamp(min=1e-9) ** (-self.idw_power)
        w = w / w.sum(dim=1, keepdim=True)
        self.register_buffer("nbr_flat", adm_flat[nn_idx])
        self.register_buffer("nbr_w", w)

    def forward(self, node_data: CompressibleNodeData) -> CompressibleNodeData:
        if self.bnd_flat.numel() == 0:
            return node_data
        m = node_data.moments
        d = node_data.distributions
        f = d.vel_old_population
        g = d.temp_old_population
        Q = f.shape[0]
        W = self.weights
        cvD = 2.0 * self.cv - self.dimension
        d_w = self.d_w
        s = d_w / (d_w + 1.0)

        idx, w = self.nbr_flat, self.nbr_w
        rho_n = m.density[..., 0].reshape(-1)[idx]
        T_n = m.temperature[..., 0].reshape(-1)[idx]
        T_ref = (w * T_n).sum(1)
        ux_ref = (w * m.velocity[0, ..., 0].reshape(-1)[idx]).sum(1)
        uy_ref = (w * m.velocity[1, ..., 0].reshape(-1)[idx]).sum(1)

        u_bnd = torch.stack([
            self.wall_velocity[0] + (ux_ref - self.wall_velocity[0]) * s,
            self.wall_velocity[1] + (uy_ref - self.wall_velocity[1]) * s,
        ], dim=0)
        if self.wall_temperature is None:
            T_bnd = T_ref.clamp(min=1e-9)
        else:
            T_bnd = (self.wall_temperature + (T_ref - self.wall_temperature) * s).clamp(min=1e-9)
        if self.density_mode == "own":
            rho_bnd = m.density[..., 0].reshape(-1)[self.bnd_flat].clamp(min=1e-9)
        else:
            p_ref = (w * (rho_n * T_n)).sum(1)
            rho_bnd = (p_ref / T_bnd).clamp(min=1e-9)

        f_bnd = W[:, None] * rho_bnd[None]
        g_bnd = cvD * T_bnd[None] * f_bnd

        f2 = f[..., 0].reshape(Q, -1)
        g2 = g[..., 0].reshape(Q, -1)
        f2[:, self.bnd_flat] = f_bnd
        g2[:, self.bnd_flat] = g_bnd

        rho2 = m.density[..., 0].reshape(-1)
        T2 = m.temperature[..., 0].reshape(-1)
        T2[self.bnd_flat] = T_bnd
        if self.solid_flat.numel():
            sflat = self.solid_flat
            rho_s = rho2[sflat]
            T_s = T2[sflat] if self.wall_temperature is None else torch.full_like(rho_s, self.wall_temperature)
            f2[:, sflat] = W[:, None] * rho_s[None]
            g2[:, sflat] = cvD * T_s[None] * (W[:, None] * rho_s[None])
            T2[sflat] = T_s

        d.vel_old_population = f2.reshape(f.shape)
        d.temp_old_population = g2.reshape(g.shape)
        m.temperature = T2.reshape(m.temperature.shape)
        for comp in range(2):
            uc = m.velocity[comp, ..., 0].reshape(-1)
            uc[self.bnd_flat] = u_bnd[comp]
            if self.solid_flat.numel():
                uc[self.solid_flat] = self.wall_velocity[comp]
            m.velocity[comp] = uc.reshape(m.velocity[comp].shape)
        e2 = m.energy[..., 0].reshape(-1)
        e2[self.bnd_flat] = self.cv * T_bnd + 0.5 * (u_bnd * u_bnd).sum(0)
        if self.solid_flat.numel():
            e2[self.solid_flat] = self.cv * T2[self.solid_flat] + 0.5 * float((self.wall_velocity ** 2).sum())
        m.energy = e2.reshape(m.energy.shape)
        return node_data

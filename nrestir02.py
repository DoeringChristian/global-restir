from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
import numpy as np
from drjitstruct import drjitstruct

from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")

from hashgrid import HashGrid


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


def p_hat(f):
    return dr.norm(f)


@drjitstruct
class RestirSample:
    Li: mi.Vector3f
    x0: mi.Point3f
    n0: mi.Vector3f
    x1: mi.Point3f
    pq: mi.Float


@drjitstruct
class RestirReservoir:
    z: RestirSample
    w: mi.Float
    W: mi.Float
    M: mi.UInt

    def update(
        self,
        sampler: mi.Sampler,
        snew: RestirSample,
        wnew: mi.Float,
        active: mi.Bool = True,
    ):
        active = mi.Bool(active)
        if dr.shape(active)[-1] == 1:
            dr.make_opaque(active)

        self.w += dr.select(active, wnew, 0)
        self.M += dr.select(active, 1, 0)
        self.z: RestirSample = dr.select(
            active & (sampler.next_1d() < wnew / self.w), snew, self.z
        )

    def merge(
        self, sampler: mi.Sampler, r: "RestirReservoir", p, active: mi.Bool = True
    ):
        active = mi.Bool(active)
        M0 = mi.UInt(self.M)
        self.update(sampler, r.z, p * r.W * r.M, active)
        self.M = dr.select(active, M0 + r.M, M0)


from tinycudann import Encoding as NGPEncoding


class NRField(nn.Module):
    def __init__(self, scene: mi.Scene, width=256, n_hidden=8) -> None:
        """Initialize an instance of NRField.

        Args:
            bb_min (mi.ScalarBoundingBox3f): minimum point of the bounding box
            bb_max (mi.ScalarBoundingBox3f): maximum point of the bounding box
        """
        super().__init__()
        self.bbox = scene.bbox()

        enc_config = {
            "otype": "HashGrid",
            # "type": "Hash",
            "base_resolution": 16,
            "n_levels": 8,
            "n_features_per_level": 4,
            "log2_hashmap_size": 22,
        }
        self.pos_enc = NGPEncoding(3, enc_config)

        in_size = 3 * 4 + self.pos_enc.n_output_dims

        hidden_layers = []
        for _ in range(n_hidden):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.ReLU(inplace=True))

        self.network = nn.Sequential(
            nn.Linear(in_size, width),
            nn.ReLU(inplace=True),
            *hidden_layers,
            nn.Linear(width, 3),
        ).to("cuda")

    def forward(self, si: mi.SurfaceInteraction3f):
        """Forward pass for NRField.

        Args:
            si (mitsuba.SurfaceInteraction3f): surface interaction
            bsdf (mitsuba.BSDF): bidirectional scattering distribution function

        Returns:
            torch.Tensor
        """
        with dr.suspend_grad():
            x = ((si.p - self.bbox.min) / (self.bbox.max - self.bbox.min)).torch()
            wi = si.to_world(si.wi).torch()
            n = si.sh_frame.n.torch()
            f_d = si.bsdf().eval_diffuse_reflectance(si).torch()

        z_x = self.pos_enc(x)

        inp = torch.concat([x, wi, n, f_d, z_x], dim=1)
        out = self.network(inp)
        out = torch.abs(out)
        return out.to(torch.float32)


class GReSTIRIntegrator(mi.SamplingIntegrator):
    search_radius = 0.1
    angle_threshold = 25 * dr.pi / 180
    temporal_M_max = 500

    def __init__(self, model: nn.Model):
        super().__init__(mi.Properties())

        self.max_depth = 8
        self.rr_depth = 4
        self.n = 0
        self.model = model
        self.batch_size = 2**14
        self.lr = 5e-4
        self.losses = []
        self.render_mode = "nerad"

    def similar(self, s1: RestirSample, s2: RestirSample) -> mi.Bool:
        similar = mi.Bool(True)
        similar &= dr.dot(s1.n0, s2.n0) > dr.cos(self.angle_threshold)

        return similar

    def create_reservoirs(self, scene: mi.Scene, n: int, resolution: int = 100):
        sampler = mi.load_dict({"type": "independent"})  # type: mi.Sampler

        sampler.seed(0, n)

        m_area = []
        for shape in scene.shapes():
            m_area.append(shape.surface_area()[0])

        m_area = np.array(m_area)

        shape_sampler = mi.DiscreteDistribution(m_area)
        self.shape_sampler = shape_sampler

        shape_idx = shape_sampler.sample(sampler.next_1d())
        shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx)  # type: mi.Shape

        ps = shape.sample_position(0.5, sampler.next_2d())  # type: mi.PositionSample3f
        self.reservoirs = dr.zeros(RestirReservoir, n)  # type: RestirReservoir
        self.reservoirs.z.x0 = ps.p
        self.reservoirs.z.n0 = ps.n
        self.grid = HashGrid(ps.p, resolution, n)
        self.n_reservoirs = n
        self.resolution = resolution

    def sample_si(
        self, scene: mi.Scene, sampler: mi.Sampler
    ) -> mi.SurfaceInteraction3f:
        shape_idx = self.shape_sampler.sample(sampler.next_1d())
        shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx)  # type: mi.Shape

        ps = shape.sample_position(0, sampler.next_2d())
        si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
        si.shape = shape
        bsdf = shape.bsdf()

        sample = sampler.next_2d()
        active_two_sided = mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
        si.wi = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere(sample),
            mi.warp.square_to_uniform_hemisphere(sample),
        )
        # pdf = dr.select(
        #     active_two_sided,
        #     mi.warp.square_to_uniform_sphere_pdf(si.wi),
        #     mi.warp.square_to_uniform_hemisphere_pdf(si.wi),
        # )
        return si

    def sample_ray(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.Ray3f,
        active: bool = True,
    ) -> mi.Color3f:
        # --------------------- Configure loop state ----------------------

        ray = mi.Ray3f(ray)
        active = mi.Bool(active)
        throughput = mi.Spectrum(1.0)
        result = mi.Spectrum(0.0)
        eta = mi.Float(1.0)
        depth = mi.UInt32(0)

        valid_ray = mi.Bool(scene.environment() is not None)

        # Variables caching information from the previous bounce
        prev_si: mi.SurfaceInteraction3f = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        bsdf_ctx = mi.BSDFContext()

        loop = mi.Loop(
            "Path Tracer",
            state=lambda: (
                sampler,
                ray,
                throughput,
                result,
                eta,
                depth,
                valid_ray,
                prev_si,
                prev_bsdf_pdf,
                prev_bsdf_delta,
                active,
            ),
        )

        loop.set_max_iterations(self.max_depth)

        while loop(active):
            si = scene.ray_intersect(ray)  # TODO: not necesarry in first interaction

            # ---------------------- Direct emission ----------------------

            ds = mi.DirectionSample3f(scene, si, prev_si)
            em_pdf = mi.Float(0.0)

            em_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)

            mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf)

            result = dr.fma(
                throughput,
                ds.emitter.eval(si, prev_bsdf_pdf > 0.0) * mis_bsdf,
                result,
            )

            active_next = ((depth + 1) < self.max_depth) & si.is_valid()

            bsdf: mi.BSDF = si.bsdf(ray)

            # ---------------------- Emitter sampling ----------------------

            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            ds, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em
            )

            wo = si.to_local(ds.d)

            # ------ Evaluate BSDF * cos(theta) and sample direction -------

            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()

            bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight = bsdf.eval_pdf_sample(
                bsdf_ctx, si, wo, sample1, sample2
            )

            # --------------- Emitter sampling contribution ----------------

            bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi)

            mi_em = dr.select(ds.delta, 1.0, mis_weight(ds.pdf, bsdf_pdf))

            result[active_em] = dr.fma(throughput, bsdf_val * em_weight * mi_em, result)

            # ---------------------- BSDF sampling ----------------------

            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi)

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            # ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight
            eta *= bsdf_sample.eta
            valid_ray |= (
                active
                & si.is_valid()
                & ~mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Null)
            )

            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            # -------------------- Stopping criterion ---------------------

            depth[si.is_valid()] += 1

            throughput_max = dr.max(throughput)

            rr_prop = dr.minimum(throughput_max * dr.sqr(eta), 0.95)
            rr_active = depth >= self.rr_depth
            rr_continue = sampler.next_1d() < rr_prop

            throughput[rr_active] *= dr.rcp(rr_prop)

            active = (
                active_next & (~rr_active | rr_continue) & (dr.neq(throughput_max, 0.0))
            )

        return dr.select(valid_ray, result, 0.0)

    def generate_sample(self, scene: mi.Scene, sampler: mi.Sampler) -> RestirSample:
        si = self.sample_si(scene, sampler)

        bsdf = si.bsdf()
        sample = sampler.next_2d()
        active_two_sided = mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
        wo = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere(sample),
            mi.warp.square_to_uniform_hemisphere(sample),
        )
        pq = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere_pdf(wo),
            mi.warp.square_to_uniform_hemisphere_pdf(wo),
        )

        si1 = scene.ray_intersect(
            si.spawn_ray(si.to_world(wo))
        )  # type: mi.SurfaceInteraction3f

        Li = self.sample_mlp(scene, si1, mode="drjit")
        # Li = self.sample_ray(scene, sampler, si.spawn_ray(si.to_world(wo)))

        sample = dr.zeros(RestirSample)  # type: RestirSample
        sample.Li = Li
        sample.pq = pq
        sample.x0 = si.p
        sample.x1 = si1.p
        sample.n0 = si.n
        return sample

    def temporal_resampling(self, scene: mi.Scene, n):
        sampler = mi.load_dict({"type": "independent"})  # type: mi.Sampler
        sampler.seed(n, self.batch_size)

        new_sample = self.generate_sample(scene, sampler)

        cell = self.grid.cell_idx(new_sample.x0)
        cell_size = dr.gather(mi.UInt, self.grid.cell_size, cell)

        index_in_cell = mi.UInt(dr.floor(sampler.next_1d() * cell_size))
        reservoir_idx = self.grid.sample_idx_in_cell(cell, index_in_cell)

        R = dr.gather(
            RestirReservoir, self.reservoirs, reservoir_idx
        )  # type: RestirReservoir

        Rnew = dr.zeros(RestirReservoir)  # type: RestirReservoir
        w = dr.select(new_sample.pq > 0, p_hat(new_sample.Li) / new_sample.pq, 0.0)
        Rnew.update(sampler, new_sample, w)

        similar = self.similar(R.z, new_sample)
        Rnew.merge(sampler, R, p_hat(R.z.Li), similar)

        Rnew.z.x0 = R.z.x0
        Rnew.z.n0 = R.z.n0

        phat = p_hat(Rnew.z.Li)
        Rnew.W = dr.select(phat * Rnew.M > 0, Rnew.w / (Rnew.M * phat), 0)
        Rnew.M = dr.clamp(Rnew.M, 0, self.temporal_M_max)

        dr.scatter(self.reservoirs, Rnew, reservoir_idx, similar)

    def sample_restir(
        self, scene: mi.Scene, sampler: mi.Sampler, si: mi.SurfaceInteraction3f
    ) -> mi.Color3f:
        Rnew = dr.zeros(RestirReservoir)  # type: RestirReservoir

        # si = scene.ray_intersect(ray)  # type: mi.SurfaceInteraction3f
        S = dr.zeros(RestirSample)  # type: RestirSample
        S.x0 = si.p
        S.n0 = si.n

        for i in range(10):
            # offset = mi.warp.square_to_uniform_disk(sampler.next_1d()) * 0.1
            offset = (
                mi.warp.square_to_uniform_disk(sampler.next_2d()) * self.search_radius
            )
            # offset = mi.warp.square_to_tent(sampler.next_2d()) * 0.01
            p = si.p + si.to_world(mi.Point3f(offset.x, offset.y, 0))

            cell = self.grid.cell_idx(p)

            cell_size = dr.gather(mi.UInt, self.grid.cell_size, cell)

            index_in_cell = mi.UInt(dr.floor(sampler.next_1d() * cell_size))
            reservoir_idx = self.grid.sample_idx_in_cell(cell, index_in_cell)

            Rn = dr.gather(
                RestirReservoir, self.reservoirs, reservoir_idx
            )  # type: RestirReservoir
            similar = self.similar(Rn.z, S)

            shadowed = scene.ray_test(si.spawn_ray_to(Rn.z.x1))

            Rnew.merge(
                sampler,
                Rn,
                p_hat(Rn.z.Li) * dr.select(~shadowed & similar, 1, 0),
                similar,
            )

        phat = p_hat(Rnew.z.Li)
        Rnew.W = dr.select(phat * Rnew.M > 0, Rnew.w / (Rnew.M * phat), 0)

        # Final sampling

        bsdf = si.bsdf()  # type: mi.BSDF
        β = bsdf.eval(mi.BSDFContext(), si, si.to_local(dr.normalize(Rnew.z.x1 - si.p)))
        emittance = si.emitter(scene).eval(si)

        result = Rnew.W * Rnew.z.Li * β + emittance
        return result

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, bool, list[float]]:
        if self.render_mode == "nerad":
            si = scene.ray_intersect(ray)
            lhs = self.sample_mlp(scene, si)

            return mi.Color3f(lhs), True, []
        elif self.render_mode == "restir":
            # self.temporal_resampling(scene, self.n)
            # self.n += 1

            si = scene.ray_intersect(ray)

            result = self.sample_restir(scene, sampler, si)

            return mi.Color3f(result), True, []

    def sample_mlp(self, scene: mi.Scene, si: mi.SurfaceInteraction3f, mode="drjit"):
        with dr.suspend_grad():
            # Le = si.emitter(scene).eval(si)

            out = self.model(si)

        if mode == "drjit":
            return mi.Spectrum(out)
        elif mode == "torch":
            return out

    def train(self, scene: mi.Scene, steps: int, debug=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        tqdm_iterator = tqdm(range(steps))

        self.model.train()
        for step in tqdm_iterator:
            # if step % 100 == 0:
            #     self.create_reservoirs(scene, self.n_reservoirs)
            with dr.suspend_grad():
                self.temporal_resampling(scene, step)

            optimizer.zero_grad()

            sampler = mi.load_dict({"type": "independent"})
            sampler.seed(step, self.batch_size)

            si = self.sample_si(scene, sampler)

            lhs = self.sample_mlp(scene, si, mode="torch")
            rhs = self.sample_restir(scene, sampler, si).torch()  # type: torch.Tensor

            rhs = rhs.reshape(self.batch_size, 3)

            loss = torch.nn.MSELoss()(lhs, rhs)
            loss.backward()
            optimizer.step()

            tqdm_iterator.set_description(f"Loos {loss.item():04f}")
            self.losses.append(loss.item())

        self.model.eval()
        # torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.manual_seed(0)
    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"] = 256
    scene["sensor"]["film"]["height"] = 256
    scene["sensor"]["film"]["rfilter"] = mi.load_dict({"type": "box"})
    scene = mi.load_dict(scene)  # type: mi.Scene
    # scene = mi.load_file("./data/scenes/living-room-3/scene.xml")

    field = NRField(scene, n_hidden=3, width=256)
    integrator = GReSTIRIntegrator(field)
    print("Creating Reservoir:")
    integrator.create_reservoirs(scene, 1_000_000)

    integrator.train(scene, 1000)

    nerad = mi.render(scene, integrator=integrator, spp=1, seed=0)
    mi.util.write_bitmap("out/nerad.png", nerad)

    integrator.render_mode = "restir"
    restir = mi.render(scene, integrator=integrator, spp=1, seed=0)
    mi.util.write_bitmap("out/restir.png", restir)

    ref = mi.render(scene, spp=8)
    mi.util.write_bitmap("out/ref.png", ref)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    fig.patch.set_visible(False)  # Hide the figure's background
    ax[0][0].axis("off")  # Remove the axes from the image
    ax[0][0].imshow(mi.util.convert_to_bitmap(nerad))
    ax[0][1].axis("off")
    ax[0][1].imshow(mi.util.convert_to_bitmap(restir))
    ax[1][0].axis("off")
    ax[1][0].imshow(mi.util.convert_to_bitmap(ref))
    ax[1][1].plot(integrator.losses, color="red")
    fig.tight_layout()  # Remove any extra white spaces around the image

    plt.show()

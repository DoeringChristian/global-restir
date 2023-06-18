from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
import numpy as np
from drjitstruct import drjitstruct

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


class GReSTIRIntegrator(mi.SamplingIntegrator):
    dist_threshold = 0.02
    angle_threshold = 25 * dr.pi / 180
    temporal_M_max = 30

    def __init__(self):
        self.max_depth = 8
        self.rr_depth = 4
        self.n = 0
        super().__init__(mi.Properties())

    def similar(self, s1: RestirSample, s2: RestirSample) -> mi.Bool:
        similar = mi.Bool(True)
        dist = dr.norm(s1.x0 - s2.x0)
        similar &= dist < self.dist_threshold
        # similar &= self.grid.same_cell(s1.x0, s2.x0)
        similar &= dr.dot(s1.n0, s2.n0) > dr.cos(self.angle_threshold)

        return similar

    def create_reservoirs(self, scene: mi.Scene, n: int):
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
        self.grid = HashGrid(ps.p, 100, n)
        self.n_reservoirs = n

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
        shape_idx = self.shape_sampler.sample(sampler.next_1d())
        shape = dr.gather(mi.ShapePtr, scene.shapes_dr(), shape_idx)  # type: mi.Shape

        ps = shape.sample_position(0, sampler.next_2d())
        si = mi.SurfaceInteraction3f(ps, dr.zeros(mi.Color0f))
        si.shape = shape
        bsdf = shape.bsdf()

        sample = sampler.next_2d()
        active_two_sided = mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide)
        wi = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere(sample),
            mi.warp.square_to_uniform_hemisphere(sample),
        )
        pq = dr.select(
            active_two_sided,
            mi.warp.square_to_uniform_sphere_pdf(wi),
            mi.warp.square_to_uniform_hemisphere_pdf(wi),
        )

        si1 = scene.ray_intersect(
            si.spawn_ray(si.to_world(wi))
        )  # type: mi.SurfaceInteraction3f

        Li = self.sample_ray(scene, sampler, si.spawn_ray(si.to_world(wi)))

        sample = dr.zeros(RestirSample)  # type: RestirSample
        sample.Li = Li
        sample.pq = pq
        sample.x0 = si.p
        sample.x1 = si1.p
        sample.n0 = si.n
        return sample

    def temporal_resampling(self, scene: mi.Scene):
        sampler = mi.load_dict({"type": "independent"})  # type: mi.Sampler
        sampler.seed(self.n, self.n_reservoirs * 2)

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
        self.n += 1

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, bool, list[float]]:
        self.temporal_resampling(scene)

        Rnew = dr.zeros(RestirReservoir)  # type: RestirReservoir

        si = scene.ray_intersect(ray)  # type: mi.SurfaceInteraction3f
        S = dr.zeros(RestirSample)  # type: RestirSample
        S.x0 = si.p
        S.n0 = si.n

        for i in range(10):
            # offset = mi.warp.square_to_uniform_disk(sampler.next_1d()) * 0.1
            offset = (
                mi.warp.square_to_std_normal(sampler.next_1d()) * self.dist_threshold
            )
            # offset = mi.warp.square_to_tent(sampler.next_2d()) * 0.01
            p = si.p + si.to_world(mi.Point3f(offset.x, offset.y, 0))

            cell = self.grid.cell_idx(p)

            cell_size = dr.gather(mi.UInt, self.grid.cell_size, cell)

            index_in_cell = mi.UInt(dr.floor(sampler.next_1d() * cell_size))
            reservoir_idx = self.grid.sample_idx_in_cell(cell, index_in_cell)

            R = dr.gather(
                RestirReservoir, self.reservoirs, reservoir_idx
            )  # type: RestirReservoir
            similar = self.similar(R.z, S)

            Rnew.merge(sampler, R, p_hat(R.z.Li), similar)

        phat = p_hat(Rnew.z.Li)
        Rnew.W = dr.select(phat * Rnew.M > 0, Rnew.w / (Rnew.M * phat), 0)

        # Final sampling

        bsdf = si.bsdf()  # type: mi.BSDF
        β = bsdf.eval(mi.BSDFContext(), si, si.to_local(dr.normalize(Rnew.z.x1 - si.p)))
        emittance = si.emitter(scene).eval(si)

        result = Rnew.W * Rnew.z.Li * β + emittance

        return mi.Color3f(result), True, []


if __name__ == "__main__":
    scene = mi.cornell_box()
    scene["sensor"]["film"]["width"] = 1024
    scene["sensor"]["film"]["height"] = 1024
    scene["sensor"]["film"]["rfilter"] = mi.load_dict({"type": "box"})
    scene = mi.load_dict(scene)  # type: mi.Scene
    scene = mi.load_file("./data/scenes/staircase/scene.xml")

    integrator = GReSTIRIntegrator()
    print("Creating Reservoir:")
    integrator.create_reservoirs(scene, 1_000_000)
    # integrator.create_reservoirs((0.01, 0.01))

    print("Rendering Images:")
    for i in range(200):
        img = mi.render(scene, integrator=integrator, spp=1, seed=i)
        mi.util.write_bitmap(f"out/{i}.png", img)
    # plt.imshow(mi.util.convert_to_bitmap(img))
    # plt.show()

    ref = mi.render(scene, spp=8)
    # plt.imshow(mi.util.convert_to_bitmap(ref))
    # plt.show()

    mi.util.write_bitmap("out/ref.png", ref)

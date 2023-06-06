from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")
    # dr.set_log_level(dr.LogLevel.Trace)

import restir
import reductions


def mis_weight(pdf_a: mi.Float, pdf_b: mi.Float) -> mi.Float:
    """
    Compute the Multiple Importance Sampling (MIS) weight given the densities
    of two sampling strategies according to the power heuristic.
    """
    a2 = dr.sqr(pdf_a)
    return dr.detach(dr.select(pdf_a > 0, a2 / dr.fma(pdf_b, pdf_b, a2), 0), True)


def p_hat(f):
    return dr.norm(f)


class GReSTIRIntegrator(mi.SamplingIntegrator):
    def __init__(self):
        self.max_depth = 8
        self.rr_depth = 4
        super().__init__(mi.Properties())

    def create_reservoirs(self, resolution: tuple[float, float]):
        width = 1.0 / resolution[0]
        height = 1.0 / resolution[1]
        self.resolution = resolution
        self.reservoir_size = mi.Vector2u(int(width), int(height))
        self.temporal = dr.zeros(restir.Reservoir, int(width * height))

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

    def sample(
        self,
        scene: mi.Scene,
        sampler: mi.Sampler,
        ray: mi.RayDifferential3f,
        medium: mi.Medium = None,
        active: bool = True,
    ) -> tuple[mi.Color3f, bool, list[float]]:
        bsdf_ctx = mi.BSDFContext()

        si = scene.ray_intersect(ray, active)  # type: mi.SurfaceInteraction3f
        active &= si.is_valid()
        bsdf: mi.BSDF = si.bsdf(ray)

        reservoir_pos = si.uv / mi.Vector2f(self.resolution[0], self.resolution[1])

        reservoir_idx = (
            mi.UInt(reservoir_pos.x) + mi.UInt(reservoir_pos.y) * self.reservoir_size.x
        )

        bsdf_sample, bsdf_val = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
        )

        wo = bsdf_sample.wo
        p_q = bsdf_sample.pdf

        bsdf_val, _ = bsdf.eval_pdf(bsdf_ctx, si, wo)

        Li = self.sample_ray(
            scene, sampler, si.spawn_ray(si.to_world(wo)), active=active
        )

        w = dr.select(dr.eq(p_q, 0), 0, p_hat(Li) / p_q)

        snew = restir.Sample()
        snew.Li = Li
        snew.wo = wo

        # R = dr.gather(
        #     restir.Reservoir, self.temporal, reservoir_idx, active
        # )  # type: restir.Reservoir

        # R.update(sampler, snew, w, active)
        # phat = p_hat(R.z.Li)
        # R.W = dr.select(dr.eq(phat * R.M, 0), 0, R.w / (R.M * phat))

        # TODO: reduce/random

        # dr.scatter(self.temporal, R, reservoir_idx, active)

        def update(a: restir.Reservoir, b: restir.SampleUpdate) -> restir.Reservoir:
            a.update(b.r, b.s, b.w)
            phat = p_hat(a.z.Li)
            a.W = dr.select(dr.eq(phat * a.M, 0), 0, a.w / (a.M * phat))
            return a

        sample_update = restir.SampleUpdate()
        sample_update.s = snew
        sample_update.w = w
        sample_update.r = sampler.next_1d()

        reductions.scatter_reduce_with(
            update, self.temporal, sample_update, reservoir_idx, active
        )

        R = dr.gather(
            restir.Reservoir, self.temporal, reservoir_idx, active
        )  # type: restir.Reservoir
        result = bsdf_val * mi.Color3f(R.z.Li) * R.W + si.emitter(scene, active).eval(
            si, active
        )

        return result, True, []


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    scene = mi.load_file("./data/scenes/cornell-box/scene.xml")  # type: mi.Scene

    integrator = GReSTIRIntegrator()
    integrator.create_reservoirs((0.0005, 0.0005))
    # integrator.create_reservoirs((0.01, 0.01))

    for i in range(100):
        img = mi.render(scene, integrator=integrator, spp=1, seed=i)
        mi.util.write_bitmap(f"out/{i}.png", img)
    # plt.imshow(mi.util.convert_to_bitmap(img))
    # plt.show()

    ref = mi.render(scene, spp=8)
    # plt.imshow(mi.util.convert_to_bitmap(ref))
    # plt.show()

    mi.util.write_bitmap("out/ref.png", ref)

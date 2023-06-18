from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
from drjitstruct import drjitstruct
from dataclasses import dataclass


@drjitstruct
class RestirSample:
    wo: mi.Vector3f
    Lo: mi.Vector3f


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


@dataclass
@drjitstruct
class SampleUpdate:
    s: Sample
    r: mi.Float
    w: mi.Float

    def __init__(self):
        self.s = Sample()
        self.r = mi.Float()
        self.w = mi.Float()

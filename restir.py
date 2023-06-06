from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
from drjitstruct import drjitstruct
from dataclasses import dataclass


@dataclass
@drjitstruct
class Sample:
    wo: mi.Vector3f
    Li: mi.Color3f

    def __init__(self):
        self.wo = mi.Vector3f()
        self.L_i = mi.Vector3f()


@dataclass
@drjitstruct
class Reservoir:
    z: Sample
    w: mi.Float
    W: mi.Float
    M: mi.UInt

    def __init__(self):
        self.w = mi.Float()
        self.W = mi.Float()
        self.M = mi.Float()
        self.z = Sample()

    def update(
        self,
        sampler: mi.Sampler,
        snew: Sample,
        wnew: mi.Float,
        active: mi.Bool = True,
    ):
        active = mi.Bool(active)
        self.w += dr.select(active, wnew, 0)
        self.M += dr.select(active, 1, 0)
        self.z = dr.select(active & (sampler.next_1d() < wnew / self.w), snew, self.z)

    def merge(self, sampler: mi.Sampler, r: "Reservoir", p, active: mi.Bool = True):
        active = mi.Bool(active)
        M0 = self.M
        self.update(sampler, r.z, p * r.W * r.M, active)
        self.M = dr.select(active, M0 + r.M, M0)

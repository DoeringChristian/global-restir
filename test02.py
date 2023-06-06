from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations
import mitsuba as mi
import drjit as dr
import numpy as np


if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


if __name__ == "__main__":
    scene = mi.load_dict(mi.cornell_box())  # type: mi.Scene

    areas = []
    ds = mi.DiscreteDistribution()
    for shape in scene.shapes():
        shape = shape  # type: mi.Shape
        areas.append(shape.surface_area()[0])

    areas = np.array(areas)
    areas = areas / areas.sum()

    print(f"{areas=}")

    ...

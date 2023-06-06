from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def scatter_reduce_with(func, target, value, index, active=True):
    n_value = dr.shape(value)[-1]
    n_target = dr.shape(target)[-1]
    print(f"{n_value=}")
    print(f"{n_target=}")

    current_scatter = dr.zeros(mi.UInt, n_target)
    processed_values = dr.full(mi.Bool, False, n_value) | ~mi.Bool(active)
    i = 0

    while not dr.all(processed_values):
        dr.scatter(
            current_scatter, dr.arange(mi.UInt, n_value), index, ~processed_values
        )

        active_lane = (
            dr.eq(
                dr.arange(mi.UInt, n_value),
                dr.gather(mi.UInt, current_scatter, index),
            )
            & ~processed_values
        )

        processed_values |= active_lane

        lane_idx = dr.compress(active_lane)

        target_idx = dr.gather(mi.UInt, index, lane_idx)

        a = dr.gather(type(target), target, target_idx)
        b = dr.gather(type(value), value, lane_idx)
        res = func(a, b)
        dr.scatter(target, res, target_idx)
        # dr.eval(target)


if __name__ == "__main__":
    target = dr.zeros(mi.Float, 10)
    index = dr.arange(mi.UInt, 20) // 2
    value = dr.ones(mi.Float, 20)

    scatter_reduce_with(lambda a, b: a + b, target, value, index)
    print(f"{target=}")

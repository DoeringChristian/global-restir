from __future__ import (
    annotations as __annotations__,
)  # Delayed parsing of type annotations

import mitsuba as mi
import drjit as dr
from dataclasses import dataclass

if __name__ == "__main__":
    mi.set_variant("cuda_ad_rgb")


def hash(p: mi.Point3u | mi.Point3f, hash_size: int):
    if isinstance(p, mi.Point3f):
        p = mi.Point3u(mi.UInt(p.x), mi.UInt(p.y), mi.UInt(p.z))
        return hash(p, hash_size)
    return ((p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791)) % hash_size


class HashGrid:
    def __init__(
        self, sample: mi.Point3f, resolution: int, n_cells: None | int = None
    ) -> None:
        """Construct a Hash Grid Similar to the implementation described in

        Guillaume Boissé. 2021. WORLD-SPACE SPATIOTEMPORAL RESERVOIR
        REUSE FOR RAY-TRACED GLOBAL ILLUMINATION. In SIGGRAPH Asia
        2021 Technical Communications (SA ’21 Technical Communications), De-
        cember 14–17, 2021, Tokyo, Japan. ACM, New York, NY, USA, 4 pages.
        https://doi.org/10.1145/3478512.3488613

        Args:
            sample: Samples that should be inserted into the hash-grid
            resolution: Number of cells in each direction
            n_cells: Number of cells in the Hash Grid
        """
        n_samples = dr.shape(sample)[-1]
        if n_cells is None:
            n_cells = n_samples
        self.n_cells = n_cells
        self.n_samples = n_samples
        self.resolution = resolution
        self.bbmin = dr.minimum(
            dr.min(sample.x), dr.minimum(dr.min(sample.y), dr.min(sample.z))
        )
        self.bbmax = dr.maximum(
            dr.max(sample.x), dr.maximum(dr.max(sample.y), dr.max(sample.z))
        )
        self.bbmax = dr.maximum(
            dr.max(sample.x), dr.maximum(dr.max(sample.y), dr.max(sample.z))
        )

        from prefix_sum import prefix_sum

        cell = self.cell_idx(sample)

        cell_size = dr.zeros(mi.UInt, n_cells)
        index_in_cell = mi.UInt(0)
        processing = dr.zeros(mi.UInt, n_cells)
        queued = mi.Bool(True)

        while dr.any(queued):
            dr.scatter(processing, dr.arange(mi.UInt, n_samples), cell, active=queued)
            selected = (
                dr.eq(
                    dr.gather(mi.UInt, processing, cell, queued),
                    dr.arange(mi.UInt, n_samples),
                )
                & queued
            )
            index_in_cell[selected] = dr.gather(mi.UInt, cell_size, cell, selected)
            dr.scatter(cell_size, index_in_cell + 1, cell, selected)
            queued &= ~selected

        first_cell = dr.eq(dr.arange(mi.UInt, n_cells), 0)
        cell_offset = prefix_sum(cell_size)
        cell_offset = dr.select(
            first_cell,
            0,
            dr.gather(
                mi.UInt,
                cell_offset,
                dr.arange(mi.UInt, n_cells) - 1,
                ~first_cell,
            ),
        )
        self.cell_offset = cell_offset
        self.cell_size = cell_size
        self.sample_idx = dr.zeros(mi.UInt, n_samples)
        dr.scatter(
            self.sample_idx,
            dr.arange(mi.UInt, n_samples),
            dr.gather(mi.UInt, cell_offset, cell) + index_in_cell,
        )

    def cell_idx(self, p: mi.Point3f):
        return hash(
            self.cell_pos(p),
            self.n_cells,
        )

    def cell_pos(self, p: mi.Point3f):
        p = (p - self.bbmin) / (self.bbmax - self.bbmin) * self.resolution
        co = mi.Point3u(mi.UInt(p.x), mi.UInt(p.y), mi.UInt(p.z))
        return co

    def same_cell(self, a: mi.Point3f, b: mi.Point3f) -> mi.Bool:
        a = self.cell_pos(a)
        b = self.cell_pos(b)
        return dr.eq(a.x, b.x) & dr.eq(a.y, b.y) & dr.eq(a.z, b.z)

    def sample_idx_in_cell(self, cell: mi.UInt, index_in_cell: mi.UInt) -> mi.UInt:
        offset = dr.gather(mi.UInt, self.cell_offset, cell)
        idx = dr.gather(mi.UInt, self.sample_idx, offset + index_in_cell)
        return idx


if __name__ == "__main__":
    # x = mi.Float(0, 0.1, 0.6, 1)
    # y = mi.Float(0, 0.1, 0.6, 1)
    # z = mi.Float(0, 0.1, 0.6, 1)
    #
    # grid = HashGrid(mi.Point3f(x, y, z), 2, 2)

    n = 20
    sampler = mi.load_dict({"type": "independent"})  # type: mi.Sampler
    sampler.seed(0, n)
    sample = mi.Point3f(sampler.next_1d(), sampler.next_1d(), sampler.next_1d())

    grid = HashGrid(sample, 3, 10)

    p = mi.Point3f(0.5, 0.5, 0.1)
    cell = grid.cell_idx(p)
    cell_size = dr.gather(mi.UInt, grid.cell_size, cell)
    offset = dr.gather(mi.UInt, grid.cell_offset, cell)
    idx = dr.gather(mi.UInt, grid.sample_idx, offset)
    x0 = dr.gather(mi.Point3f, sample, idx)
    print(f"{x0=}")

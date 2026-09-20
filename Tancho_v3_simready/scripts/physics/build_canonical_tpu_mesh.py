#!/usr/bin/env python3
"""Build a wheel-axis-centred TPU collision STL from the canonical source mesh."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    data = bytearray(args.source.read_bytes())
    if len(data) < 84:
        raise ValueError("STL is too short")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + triangle_count * 50:
        raise ValueError("Only binary STL input is supported")

    vertices = np.empty((triangle_count * 3, 3), dtype=np.float64)
    for index in range(triangle_count):
        vertices[index * 3 : (index + 1) * 3] = np.frombuffer(
            data, dtype="<f4", count=9, offset=84 + index * 50 + 12
        ).reshape(3, 3)

    # Wheel spin is local Z.  The CAD exports place the left TPU thickness at
    # z=[-1.5, 15.5] mm, so centre only that axial extent.  X/Y remain untouched
    # to preserve the original support profile and wheel phase exactly.
    axial_offset = 0.5 * (vertices[:, 2].min() + vertices[:, 2].max())
    for index in range(triangle_count):
        offset = 84 + index * 50 + 12
        triangle = np.frombuffer(data, dtype="<f4", count=9, offset=offset).reshape(3, 3).copy()
        triangle[:, 2] -= axial_offset
        data[offset : offset + 36] = triangle.astype("<f4").tobytes()

    header = b"Tancho canonical TPU collision; wheel axis centered at local z=0"
    data[:80] = header.ljust(80, b" ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(data)
    print(f"SOURCE_AXIAL_OFFSET_M={axial_offset:.12g}")
    print(f"OUTPUT={args.output.resolve()}")


if __name__ == "__main__":
    main()

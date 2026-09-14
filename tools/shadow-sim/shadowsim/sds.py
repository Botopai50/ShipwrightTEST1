"""SDS1 reader.

The format is defined by WriteShadowDepthCapture in
libultraship/include/fast/backends/shadow_capture.h:

    u32 magic ('SDS1', little-endian)  u32 width  u32 height  u32 slices
    per slice, per row:  u32 run_count, then run_count x (u16 depth, u32 length)

Depths are the raw D16_UNORM bits, so the shader's normalised depth is
value / 65535. Nothing here converts: the comparison the kernel performs is
against the stored value, and a lossy step on the way in would be a
difference between this reproduction and the frame it reproduces.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = 0x31534453
SCENE_MAGIC = 0x315A4453  # "SDZ1": the scene depth, 24-bit
_HEADER = struct.Struct("<4I")
_RUNS = struct.Struct("<I")
_RUN = struct.Struct("<HI")


class SdsError(ValueError):
    """The file is not a well-formed SDS1 capture."""


def load(path: str | Path) -> np.ndarray:
    """Return the depth slices as uint16 of shape (slices, height, width)."""
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER.size:
        raise SdsError(f"{path}: too short to hold an SDS1 header")
    magic, width, height, slices = _HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise SdsError(f"{path}: magic {magic:#010x}, expected {MAGIC:#010x}")
    if not (0 < width <= 8192 and 0 < height <= 8192 and 0 < slices <= 64):
        raise SdsError(f"{path}: implausible dimensions {width}x{height}x{slices}")

    out = np.empty((slices, height, width), np.uint16)
    offset = _HEADER.size
    for s in range(slices):
        for y in range(height):
            try:
                (runs,) = _RUNS.unpack_from(raw, offset)
            except struct.error as exc:
                raise SdsError(f"{path}: truncated at slice {s} row {y}") from exc
            offset += _RUNS.size
            # A row is width texels however it is encoded, so the run lengths must
            # sum to exactly that. Checking here is what turns a desynchronised
            # stream into an error naming the row, instead of a silently shifted
            # image that still decodes to the end.
            if runs == 0 or runs > width:
                raise SdsError(f"{path}: slice {s} row {y} claims {runs} runs for {width} texels")
            row = out[s, y]
            x = 0
            for _ in range(runs):
                try:
                    depth, length = _RUN.unpack_from(raw, offset)
                except struct.error as exc:
                    raise SdsError(f"{path}: truncated in slice {s} row {y}") from exc
                offset += _RUN.size
                if length == 0 or x + length > width:
                    raise SdsError(f"{path}: slice {s} row {y} run overruns the row")
                row[x : x + length] = depth
                x += length
            if x != width:
                raise SdsError(f"{path}: slice {s} row {y} covered {x} of {width} texels")
    if offset != len(raw):
        raise SdsError(f"{path}: {len(raw) - offset} trailing bytes after {slices} slices")
    return out


def save(path: str | Path, depths: np.ndarray) -> None:
    """Write depths (slices, height, width) back out as SDS1.

    Round-tripping is what lets the tests assert on the reader without a GPU:
    a file this writes and the reader reads must give back the same array.
    """
    depths = np.ascontiguousarray(depths, np.uint16)
    if depths.ndim != 3:
        raise SdsError(f"expected (slices, height, width), got {depths.shape}")
    slices, height, width = depths.shape
    chunks = [_HEADER.pack(MAGIC, width, height, slices)]
    for s in range(slices):
        for y in range(height):
            row = depths[s, y]
            # Run starts: the first texel, plus every texel differing from its
            # predecessor. Same rule the writer uses, expressed as a mask.
            starts = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
            lengths = np.diff(np.r_[starts, width])
            chunks.append(_RUNS.pack(len(starts)))
            chunks.extend(_RUN.pack(int(d), int(n)) for d, n in zip(row[starts], lengths))
    Path(path).write_bytes(b"".join(chunks))


_SCENE_RUN = struct.Struct("<II")


def load_scene(path: str | Path) -> np.ndarray:
    """Read an SDZ1 scene depth buffer as uint32 of shape (height, width).

    Written by WriteSceneDepthCapture. Same header and same per-row run encoding
    as SDS1, but the depth of a run is u32 holding the D24_UNORM value -- so the
    normalised depth is value / 16777215, not / 65535. The two formats are kept
    apart by magic rather than by a flag: a reader that guessed wrong would be
    off by 256x and still produce a plausible-looking image.
    """
    raw = Path(path).read_bytes()
    if len(raw) < _HEADER.size:
        raise SdsError(f"{path}: too short to hold an SDZ1 header")
    magic, width, height, slices = _HEADER.unpack_from(raw, 0)
    if magic != SCENE_MAGIC:
        raise SdsError(f"{path}: magic {magic:#010x}, expected {SCENE_MAGIC:#010x}")
    if not (0 < width <= 8192 and 0 < height <= 8192 and slices == 1):
        raise SdsError(f"{path}: implausible dimensions {width}x{height}x{slices}")

    out = np.empty((height, width), np.uint32)
    offset = _HEADER.size
    for y in range(height):
        (runs,) = _RUNS.unpack_from(raw, offset)
        offset += _RUNS.size
        if runs == 0 or runs > width:
            raise SdsError(f"{path}: row {y} claims {runs} runs for {width} texels")
        row = out[y]
        x = 0
        for _ in range(runs):
            depth, length = _SCENE_RUN.unpack_from(raw, offset)
            offset += _SCENE_RUN.size
            if length == 0 or x + length > width:
                raise SdsError(f"{path}: row {y} run overruns the row")
            row[x : x + length] = depth
            x += length
        if x != width:
            raise SdsError(f"{path}: row {y} covered {x} of {width} texels")
    if offset != len(raw):
        raise SdsError(f"{path}: {len(raw) - offset} trailing bytes")
    return out

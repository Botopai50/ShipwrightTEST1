"""A capture on disk: capture.json plus the depth slices it names.

The field names here are the names in the HLSL cbuffer (PerShadowCB in
libultraship/src/fast/shaders/directx/default.shader.hlsl), not tidier ones.
A reproduction that renames its inputs cannot be read side by side with the
shader it is meant to reproduce, and that comparison is the whole point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import sds

# The shader's own limits, mirrored from fast/shadow_map.h.
MAX_CASCADES = 3


class CaptureError(ValueError):
    """The capture is missing something the kernel needs."""


@dataclass
class ShadowParams:
    """Every float4 the receiver reads, straight out of capture.json.

    The capture writer carries a static_assert that PerShadowCB gained no field
    without a matching CAPTURE_FIELD; `require` below is the same idea on this
    side, so a capture written by an older build fails by name instead of
    reproducing with a silently defaulted input.
    """

    view_proj: np.ndarray  # (cascades, 4, 4) row-major, row-vector
    splits: np.ndarray
    texel_world: np.ndarray
    texel_uv: np.ndarray
    params: np.ndarray  # x=count y=fade band z=- w=darkness
    range_: np.ndarray  # x=furthest footprint y=debug view
    edge: np.ndarray  # y=analytic width z=jitter on w=taps
    jitter: np.ndarray  # x=radius in texels
    acne0: np.ndarray  # x=move on y=texels
    acne1: np.ndarray  # x=slope on y=slope cap z=acne master
    harden: np.ndarray  # x=on y=hardness z=threshold
    actor_min: np.ndarray
    actor_max: np.ndarray
    actor_texel_uv: np.ndarray
    slice_valid: list[bool] = field(default_factory=list)

    @property
    def cascade_count(self) -> int:
        return int(self.params[0])

    @property
    def light_axis(self) -> np.ndarray:
        """ShadowLightAxis(): normalize(shadow_view_proj[0]._13_23_33).

        The projection is orthographic and stored row-major for a row-vector
        multiply, so the z output column is the light direction, and its length
        is the world-to-depth scale (see ShadowDepthScaleAt).
        """
        axis = self.view_proj[0][:3, 2]
        return axis / np.linalg.norm(axis)

    def depth_scale(self, cascade: int) -> float:
        return float(np.linalg.norm(self.view_proj[cascade][:3, 2]))


def _vec4(meta: dict, key: str) -> np.ndarray:
    if key not in meta:
        raise CaptureError(
            f"capture.json has no {key!r}. It was written by a build whose "
            f"PerShadowCB predates this field; re-capture, or the kernel would "
            f"run on a default this frame never used."
        )
    return np.asarray(meta[key], np.float64)


@dataclass
class Capture:
    directory: Path
    meta: dict
    world: np.ndarray  # (slices, h, w) uint16
    actors: np.ndarray | None
    params: ShadowParams

    @classmethod
    def load(cls, directory: str | Path) -> "Capture":
        directory = Path(directory)
        meta_path = directory / "capture.json"
        if not meta_path.is_file():
            raise CaptureError(f"{directory}: no capture.json")
        meta = json.loads(meta_path.read_text())

        fmt = meta.get("format")
        if fmt != "SDS1":
            raise CaptureError(f"{directory}: format {fmt!r}, this reads SDS1")
        # `complete` is written last, after both depth files have closed. A
        # capture without it was interrupted, and its slices may be short.
        if not meta.get("complete"):
            raise CaptureError(f"{directory}: capture.json is not marked complete")

        world = sds.load(directory / "world.sds")
        actors = None
        actor_layer = meta.get("actor_layer")
        if actor_layer:
            actors = sds.load(directory / actor_layer)

        matrices = meta.get("slice_matrices") or []
        if not matrices:
            raise CaptureError(f"{directory}: capture.json has no slice_matrices")
        view_proj = np.asarray(matrices, np.float64).reshape(len(matrices), 4, 4)

        params = ShadowParams(
            view_proj=view_proj,
            splits=_vec4(meta, "shadow_splits"),
            texel_world=_vec4(meta, "shadow_texel_world"),
            texel_uv=_vec4(meta, "shadow_texel_uv"),
            params=_vec4(meta, "shadow_params"),
            range_=_vec4(meta, "shadow_range"),
            edge=_vec4(meta, "shadow_edge"),
            jitter=_vec4(meta, "shadow_jitter"),
            acne0=_vec4(meta, "shadow_acne0"),
            acne1=_vec4(meta, "shadow_acne1"),
            harden=_vec4(meta, "shadow_harden"),
            actor_min=_vec4(meta, "shadow_actor_min"),
            actor_max=_vec4(meta, "shadow_actor_max"),
            actor_texel_uv=_vec4(meta, "shadow_actor_texel_uv"),
            slice_valid=list(meta.get("slice_valid") or []),
        )

        count = params.cascade_count
        if not 0 < count <= len(view_proj):
            raise CaptureError(
                f"{directory}: shadow_params.x says {count} cascades but "
                f"{len(view_proj)} matrices were written"
            )
        if world.shape[0] < count:
            raise CaptureError(
                f"{directory}: world.sds holds {world.shape[0]} slices, "
                f"fewer than the {count} cascades the frame used"
            )
        return cls(directory, meta, world, actors, params)

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.world.shape[2]), int(self.world.shape[1])

    def describe(self) -> str:
        p = self.params
        w, h = self.resolution
        ctx = self.meta.get("game_context") or {}
        lines = [
            f"{self.directory}",
            f"  world   {p.cascade_count} cascades, {w}x{h}, valid={p.slice_valid}",
            f"  actors  {0 if self.actors is None else self.actors.shape[0]} slices"
            f"  (own texture: {self.meta.get('actor_own_texture')})",
            f"  light   {np.array2string(p.light_axis, precision=4)}",
            f"  splits  {p.splits[:3]}   texel world {p.texel_world[:3]}",
            f"  acne    move={p.acne0[0] > 0.5} texels={p.acne0[1]:g}"
            f"  slope={p.acne1[0] > 0.5} cap={p.acne1[1]:g} master={p.acne1[2] > 0.5}",
            f"  edge    analytic={p.edge[0] > 0.5} width={p.edge[1]:g} jitter={p.edge[2] > 0.5}"
            f" taps={int(max(p.edge[3], 1))} radius={p.jitter[0]:g} texels",
            f"  harden  on={p.harden[0] > 0.5} hardness={p.harden[1]:g} threshold={p.harden[2]:g}",
            f"  darkness {p.params[3]:g}   fade band {p.params[1]:g}",
        ]
        if ctx:
            lines.append("  context " + ", ".join(f"{k}={v}" for k, v in sorted(ctx.items())))
        return "\n".join(lines)

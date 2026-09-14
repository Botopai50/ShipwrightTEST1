"""Turn the depth slices back into the surface they were rasterised from.

A shadow map is a depth image of the scene as the light sees it, so every texel
that holds anything is a real surface point and the whole slice unprojects to
the caster geometry -- the castle, the walls, the ground -- in world space.

That reconstructed surface is also the RECEIVER this harness shades. It is the
same surface the depth pass wrote, which is what makes self-shadowing artefacts
(acne, and the peter-panning a bias trades it for) reproduce here rather than
merely being modelled: the comparison is run against the depths the surface
itself produced, exactly as it is in the frame.

What the capture does not carry is the camera and the colour buffer, so what is
reproduced is the shadow term on the scene's surfaces, not the final picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .capture import Capture

FAR = 65535
SCENE_FAR = 16777215.0  # D24_UNORM


@dataclass
class Surface:
    """Unprojected surface points, flat arrays indexed alike."""

    world: np.ndarray  # (N, 3)
    normal: np.ndarray  # (N, 3)
    texel: np.ndarray  # (N, 2) the texel each point came from
    cascade: int
    shape: tuple[int, int]  # (height, width) of the slice it came from
    valid: np.ndarray  # (H, W) bool, which texels produced a point
    # Set only by `scene`, where the capture's own matrix gives the exact clip w.
    # Everywhere else the view depth is measured from a chosen eye instead.
    view_depth: np.ndarray | None = None


def unproject(capture: Capture, cascade: int, step: int = 1) -> Surface:
    """Unproject one world cascade into world-space points with normals.

    `step` subsamples the texel grid. Subsampling drops points, it does not
    filter them: each surviving point is the exact position that texel encodes,
    so a coarse run and a full one differ in how many samples are shaded, never
    in what any one of them says.
    """
    depth = capture.world[cascade]
    height, width = depth.shape
    inv = np.linalg.inv(capture.params.view_proj[cascade])

    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    raw = depth[grid_y, grid_x]

    # NDC -> world, inverting ShadowProject: uv.x = ndc.x*0.5+0.5 and
    # uv.y = -ndc.y*0.5+0.5, with uv = (texel + 0.5) / resolution.
    ndc_x = (grid_x + 0.5) / width * 2.0 - 1.0
    ndc_y = 1.0 - (grid_y + 0.5) / height * 2.0
    ndc_z = raw.astype(np.float64) / 65535.0

    flat = np.stack(
        [ndc_x.ravel(), ndc_y.ravel(), ndc_z.ravel(), np.ones(ndc_z.size)], axis=1
    )
    world = flat @ inv
    world = world[:, :3] / world[:, 3:4]
    world = world.reshape(*ndc_z.shape, 3)

    # A texel left at FAR holds no surface. Its unprojected point sits on the far
    # plane, which is not a place anything was drawn, so it is excluded from the
    # samples AND from the neighbours the normals are built from.
    valid = raw < FAR
    normal = _normals(world, valid, capture.params.light_axis)

    keep = valid & np.isfinite(normal).all(axis=2)
    idx = np.flatnonzero(keep.ravel())
    return Surface(
        world=world.reshape(-1, 3)[idx],
        normal=normal.reshape(-1, 3)[idx],
        texel=np.stack([grid_x.ravel()[idx], grid_y.ravel()[idx]], axis=1),
        cascade=cascade,
        shape=ndc_z.shape,
        valid=keep,
    )


def _normals(world: np.ndarray, valid: np.ndarray, light_axis: np.ndarray) -> np.ndarray:
    """Face normals from the reconstructed positions.

    Central differences where both neighbours hold surface, one-sided where only
    one does. A difference taken across a depth discontinuity would describe a
    surface that is not there -- and those silhouettes are exactly where shadow
    artefacts collect -- so a neighbour at FAR is never differenced against.
    """
    height, width, _ = world.shape
    du = np.zeros_like(world)
    dv = np.zeros_like(world)

    right = np.zeros((height, width), bool)
    right[:, :-1] = valid[:, :-1] & valid[:, 1:]
    left = np.zeros((height, width), bool)
    left[:, 1:] = valid[:, 1:] & valid[:, :-1]
    du[:, :-1][right[:, :-1]] = (world[:, 1:] - world[:, :-1])[right[:, :-1]]
    one_sided = left & ~right
    du[:, 1:][one_sided[:, 1:]] = (world[:, 1:] - world[:, :-1])[one_sided[:, 1:]]

    down = np.zeros((height, width), bool)
    down[:-1, :] = valid[:-1, :] & valid[1:, :]
    up = np.zeros((height, width), bool)
    up[1:, :] = valid[1:, :] & valid[:-1, :]
    dv[:-1, :][down[:-1, :]] = (world[1:, :] - world[:-1, :])[down[:-1, :]]
    one_sided = up & ~down
    dv[1:, :][one_sided[1:, :]] = (world[1:, :] - world[:-1, :])[one_sided[1:, :]]

    normal = np.cross(du, dv)
    length = np.linalg.norm(normal, axis=2, keepdims=True)
    # Points with no usable neighbour get the light axis back, so they read as
    # fully facing the light and contribute no invented slope.
    degenerate = (length < 1e-12).squeeze(-1)
    normal = np.divide(normal, np.maximum(length, 1e-12))
    normal[degenerate] = -light_axis
    # Orient towards the light: the winding of the texel grid is arbitrary, and a
    # flipped normal would send the acne offset into the surface instead of off it.
    flip = (normal @ -light_axis) < 0.0
    normal[flip] *= -1.0
    return normal


def plane(
    capture: Capture,
    height_y: float = 0.0,
    centre: np.ndarray | None = None,
    extent: float = 1400.0,
    resolution: int = 512,
) -> Surface:
    """A flat receiver at a fixed height, sampled on a world-space grid.

    This is the receiver that shows the CAST shadow, and the reason it has to
    exist. Unprojecting the shadow map gives back the surface nearest the light,
    so by construction almost nothing on it is occluded -- the ground lying in
    the castle's shadow is behind the castle along the light and was never
    written. A plane is not subject to that: every grid point is projected into
    the map and compared like any receiver, so the points the castle covers come
    back occluded and the shadow is drawn.

    It is a real receiver, not a visualisation: the kernel run over it is the one
    the shader runs, with this capture's inputs. What it is not is the scene's
    own ground, which the capture does not carry -- so the shadow is exact and
    the surface it falls on is idealised.
    """
    if centre is None:
        p = capture.params
        centre = (
            (p.actor_min[:3] + p.actor_max[:3]) * 0.5
            if np.all(p.actor_min[:3] <= p.actor_max[:3])
            else np.zeros(3)
        )
    half = extent * 0.5
    axis = np.linspace(-half, half, resolution)
    gx, gz = np.meshgrid(axis, axis, indexing="xy")

    world = np.empty((resolution * resolution, 3), np.float64)
    world[:, 0] = (centre[0] + gx).ravel()
    world[:, 1] = height_y
    world[:, 2] = (centre[2] + gz).ravel()

    normal = np.zeros_like(world)
    normal[:, 1] = 1.0

    ty, tx = np.meshgrid(
        np.arange(resolution), np.arange(resolution), indexing="ij"
    )
    return Surface(
        world=world,
        normal=normal,
        texel=np.stack([tx.ravel(), ty.ravel()], axis=1),
        cascade=-1,
        shape=(resolution, resolution),
        valid=np.ones((resolution, resolution), bool),
    )


def scene(capture: Capture, step: int = 2) -> Surface:
    """Unproject the CAMERA's depth buffer: the receiver, at last.

    Every visible surface is in here, shadowed ones included, which is the whole
    difference from `unproject` above -- that one can only ever return the
    surface the light already sees, so almost nothing on it can be occluded.

    The view depth is not measured from a guessed eye: the shader's `viewDepth`
    is `input.position.w`, the clip w, and unprojecting through the inverse
    yields world_h / clip.w, so the reciprocal of the fourth component IS that w.
    Exact, and it costs nothing.
    """
    if capture.scene is None or capture.camera_view_proj is None:
        raise ValueError(
            f"{capture.directory} carries no receiver. It was taken by a build "
            f"that writes only the caster layers; re-capture, or use the plane."
        )
    depth = capture.scene
    height, width = depth.shape
    inv = np.linalg.inv(capture.camera_view_proj)

    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    raw = depth[grid_y, grid_x]

    ndc = np.stack(
        [
            (grid_x + 0.5) / width * 2.0 - 1.0,
            1.0 - (grid_y + 0.5) / height * 2.0,
            raw.astype(np.float64) / SCENE_FAR,
            np.ones(raw.shape),
        ],
        axis=-1,
    )
    homogeneous = ndc.reshape(-1, 4) @ inv
    w = homogeneous[:, 3]
    world = (homogeneous[:, :3] / w[:, None]).reshape(*raw.shape, 3)
    view_depth = (1.0 / w).reshape(raw.shape)

    valid = (raw < SCENE_FAR) & np.isfinite(world).all(axis=2) & (view_depth > 0)
    normal = _normals(world, valid, capture.params.light_axis)
    keep = valid & np.isfinite(normal).all(axis=2)
    idx = np.flatnonzero(keep.ravel())

    surface = Surface(
        world=world.reshape(-1, 3)[idx],
        normal=normal.reshape(-1, 3)[idx],
        # The jitter hash is a function of the SCREEN coordinate, and here the
        # samples are screen pixels, so this is the shader's own input.
        texel=np.stack([grid_x.ravel()[idx], grid_y.ravel()[idx]], axis=1),
        cascade=-2,
        shape=raw.shape,
        valid=keep,
    )
    surface.view_depth = view_depth.ravel()[idx]
    return surface

"""Views over a reconstructed capture.

Two of them, and the difference matters when reading a result:

* The LIGHT view shades one sample per shadow-map texel, in that map's own grid.
  Nothing is resampled and every pixel is a real surface point, so this is the
  view to read when the question is about the comparison itself -- acne, bias,
  the analytic edge, the jitter pattern, a cascade seam.

* The CAMERA view projects the same shaded points into a perspective camera, so
  the shadow can be looked at the way it is seen in the game. The surface is a
  point cloud at the light's texel density, so at the near end it is sparser than
  the screen and the splat below fills the gaps. Shapes and boundaries are
  faithful; a per-pixel comparison against a screenshot is not what it is for.

The capture carries no camera, so `eye` is supplied. It is not cosmetic: the
cascade a point lands in, and therefore its resolution and its cross-fade, are
chosen from the distance to the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kernel
from .capture import Capture
from .reconstruct import Surface, plane, unproject


@dataclass
class Shaded:
    """Per-sample results, and where each sample came from."""

    surface: Surface
    coverage: np.ndarray
    visibility: np.ndarray
    lit: np.ndarray
    cascade: np.ndarray
    view_depth: np.ndarray


def default_eye(capture: Capture) -> np.ndarray:
    """The centre of the actor box.

    That box is where the characters were, so for a capture taken with the player
    on screen it is within a few units of the camera's subject -- a far better
    stand-in for the missing camera than the origin, which for these maps can be
    thousands of units away and would put the whole scene in the coarsest cascade.
    """
    p = capture.params
    if np.all(p.actor_min[:3] <= p.actor_max[:3]):
        return (p.actor_min[:3] + p.actor_max[:3]) * 0.5
    return np.zeros(3)


def shade_surface(
    capture: Capture,
    surface: Surface,
    eye: np.ndarray,
    want_actors: bool = True,
) -> Shaded:
    """Run the receiver kernel over every point of a reconstructed surface."""
    p = capture.params
    world = surface.world
    view_depth = np.linalg.norm(world - eye, axis=1)
    # The jitter hash is a function of the pixel coordinate, so the sample's own
    # grid position is what it must be fed: feeding an index would correlate the
    # rotation with memory order and print the pattern along rows.
    pixel = surface.texel.astype(np.float64)

    lit = kernel.lit_layers(
        world, view_depth, surface.normal, pixel,
        capture.world, capture.actors, p, want_actors=want_actors,
    )
    coverage, visibility = kernel.shade(lit, p)
    return Shaded(
        surface=surface,
        coverage=coverage,
        visibility=visibility,
        lit=lit,
        cascade=kernel.cascade_index(view_depth, p),
        view_depth=view_depth,
    )


def to_image(shaded: Shaded, channel: str = "visibility") -> np.ndarray:
    """Scatter per-sample values back into the slice's texel grid.

    Texels that held no surface are left at 1.0 (lit), matching what the sampler
    returns there, and are what the empty regions of the image are.
    """
    height, width = shaded.surface.shape
    values = _channel(shaded, channel)
    if values.ndim == 1:
        image = np.ones((height, width), np.float64)
    else:
        image = np.ones((height, width, values.shape[1]), np.float64)
    flat = np.flatnonzero(shaded.surface.valid.ravel())
    image.reshape(height * width, -1)[flat] = values.reshape(len(values), -1)
    return image


_CASCADE_COLOURS = np.array([[1.0, 0.25, 0.25], [0.25, 1.0, 0.25], [0.35, 0.5, 1.0]])


def _channel(shaded: Shaded, channel: str) -> np.ndarray:
    """The debug views, matching the shader's shadow_range.y selectors."""
    if channel == "visibility":
        return shaded.visibility
    if channel == "coverage":  # debug 5
        return shaded.coverage
    if channel == "layers":  # debug 2: green = world occludes, red = actor
        out = np.ones((len(shaded.coverage), 3))
        out[:, 0] = shaded.lit[:, 1]
        out[:, 2] = shaded.lit[:, 1]
        out[:, 1] = shaded.lit[:, 0]
        out[:, 2] = np.minimum(out[:, 2], shaded.lit[:, 0])
        return out
    if channel == "cascade":  # debug 7
        return _CASCADE_COLOURS[np.minimum(shaded.cascade, 2)]
    if channel == "normal":  # debug 3
        return shaded.surface.normal * 0.5 + 0.5
    if channel == "depth":
        d = shaded.view_depth
        span = max(d.max() - d.min(), 1e-9)
        return 1.0 - (d - d.min()) / span
    raise ValueError(f"unknown channel {channel!r}")


def render_light_view(
    capture: Capture,
    cascade: int = 0,
    step: int = 2,
    eye: np.ndarray | None = None,
    want_actors: bool = True,
) -> Shaded:
    surface = unproject(capture, cascade, step=step)
    return shade_surface(capture, surface, default_eye(capture) if eye is None else eye, want_actors)


def render_plane_view(
    capture: Capture,
    height_y: float = 0.0,
    extent: float = 6000.0,
    resolution: int = 512,
    centre: np.ndarray | None = None,
    eye: np.ndarray | None = None,
    want_actors: bool = True,
) -> Shaded:
    """Shade a flat receiver: the view that shows the cast shadow itself."""
    surface = plane(capture, height_y, centre, extent, resolution)
    return shade_surface(capture, surface, default_eye(capture) if eye is None else eye, want_actors)


def receivers(
    capture: Capture,
    kind: str,
    step: int,
    cascades: tuple[int, ...],
    height_y: float,
    extent: float,
    resolution: int,
) -> list[Surface]:
    """The surfaces a view shades, by name.

    `surface` is the geometry the light saw and `plane` is the idealised ground
    it falls on; `both` draws the castle standing on its own shadow, which is
    the only one of the three that reads as a scene.
    """
    out: list[Surface] = []
    if kind in ("surface", "both"):
        out += [
            unproject(capture, c, step=step)
            for c in cascades
            if c < capture.params.cascade_count
        ]
    if kind in ("plane", "both"):
        out.append(plane(capture, height_y, None, extent, resolution))
    if not out:
        raise ValueError(f"receiver {kind!r} selected nothing to shade")
    return out


def default_camera(capture: Capture, target: np.ndarray) -> np.ndarray:
    """A viewpoint derived from the capture, since it carries no camera.

    Placed PERPENDICULAR to the light's horizontal travel and above it. Standing
    with the sun behind the camera hides every shadow behind the thing casting
    it, and standing facing the sun puts them all edge-on; across the light is
    the one arrangement where their length and their edge are both visible.
    """
    axis = capture.params.light_axis
    horizontal = np.array([axis[0], 0.0, axis[2]])
    norm = np.linalg.norm(horizontal)
    horizontal = horizontal / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    side = np.array([-horizontal[2], 0.0, horizontal[0]])
    # Sized from the middle cascade, which is the band that holds the scene's
    # bulk: the near one frames a few metres, the far one the whole map.
    distance = float(capture.params.splits[1]) * 0.9
    return target + side * distance + np.array([0.0, distance * 0.75, 0.0])


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    up = np.asarray(up, np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= max(np.linalg.norm(right), 1e-9)
    true_up = np.cross(right, forward)
    return np.stack([right, true_up, forward])


def render_camera_view(
    capture: Capture,
    eye: np.ndarray,
    target: np.ndarray,
    width: int = 960,
    height: int = 540,
    fov_y_degrees: float = 60.0,
    surfaces: list[Surface] | None = None,
    channel: str = "visibility",
    splat: int = 2,
    want_actors: bool = True,
    ground_y: float | None = 0.0,
) -> np.ndarray:
    """Render from a viewpoint: a ray-cast ground, then the geometry over it.

    The ground is intersected per pixel rather than splatted. A grid of world
    points dense enough to fill the screen at the far end is far too sparse at
    the near end, and the holes that leaves are not a property of the shadow --
    they are a property of the grid, and they read as noise in exactly the region
    the eye is drawn to. One ray per pixel has no such spacing, and the point it
    finds is where the ground really is for that pixel.

    The reconstructed geometry has no such analytic form -- it is samples -- so it
    is splatted with a depth test on top of the ground.
    """
    basis = look_at(eye, target)
    tan_half = np.tan(np.radians(fov_y_degrees) * 0.5)
    aspect = width / height

    colour = np.ones((height, width, 3), np.float64)
    zbuffer = np.full((height, width), np.inf)

    if ground_y is not None:
        _cast_ground(capture, eye, basis, tan_half, aspect, width, height,
                     ground_y, channel, want_actors, colour, zbuffer)

    for surface in surfaces or []:
        if surface.cascade < 0:
            continue  # the flat receiver is the ray-cast ground above
        shaded = shade_surface(capture, surface, eye, want_actors)
        values = _channel(shaded, channel)
        if values.ndim == 1:
            values = np.repeat(values[:, None], 3, axis=1)

        cam = (shaded.surface.world - eye) @ basis.T
        front = cam[:, 2] > 1e-4
        if not front.any():
            continue
        cam, values = cam[front], values[front]
        px = ((cam[:, 0] / (cam[:, 2] * tan_half * aspect) * 0.5 + 0.5) * width).astype(np.int64)
        py = ((0.5 - cam[:, 1] / (cam[:, 2] * tan_half) * 0.5) * height).astype(np.int64)

        for dy in range(splat):
            for dx in range(splat):
                x, y = px + dx, py + dy
                on = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                if on.any():
                    _zwrite(colour, zbuffer, x[on], y[on], cam[on, 2], values[on])
    return colour


def _cast_ground(capture, eye, basis, tan_half, aspect, width, height,
                 ground_y, channel, want_actors, colour, zbuffer):
    """One ray per pixel against the horizontal plane at ground_y."""
    xs = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    ys = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    gx, gy = np.meshgrid(xs, ys)
    dirs = (
        basis[0] * (gx * tan_half * aspect)[..., None]
        + basis[1] * (gy * tan_half)[..., None]
        + basis[2]
    )
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)

    denom = dirs[..., 1]
    # Rays parallel to the plane, and those pointing away from it, never meet it.
    hit = np.abs(denom) > 1e-9
    t = np.full(denom.shape, -1.0)
    np.divide(ground_y - eye[1], denom, out=t, where=hit)
    hit &= t > 1e-4
    if not hit.any():
        return

    idx = np.flatnonzero(hit.ravel())
    world = eye + dirs.reshape(-1, 3)[idx] * t.ravel()[idx, None]
    normal = np.zeros_like(world)
    normal[:, 1] = 1.0
    # The jitter hash wants the screen position, which here is the pixel itself.
    pixel = np.stack([idx % width, idx // width], axis=1).astype(np.float64)

    surface = Surface(world=world, normal=normal, texel=pixel.astype(np.int64),
                      cascade=-1, shape=(height, width), valid=hit)
    shaded = shade_surface(capture, surface, eye, want_actors)
    values = _channel(shaded, channel)
    if values.ndim == 1:
        values = np.repeat(values[:, None], 3, axis=1)
    colour.reshape(-1, 3)[idx] = values
    # Depth along the view axis, so the splats above test against the same scale.
    zbuffer.ravel()[idx] = ((world - eye) @ basis[2])


def _zwrite(colour, zbuffer, x, y, depth, values):
    """Scatter with a depth test.

    numpy's fancy assignment keeps the LAST write to a repeated index, not the
    nearest, so the points are sorted far-to-near first and the nearest lands
    last. Sorting is what makes this a depth test rather than a race.
    """
    order = np.argsort(-depth)
    x, y, depth, values = x[order], y[order], depth[order], values[order]
    flat = y * colour.shape[1] + x
    current = zbuffer.ravel()[flat]
    closer = depth < current
    if not closer.any():
        return
    flat, depth, values = flat[closer], depth[closer], values[closer]
    zbuffer.ravel()[flat] = depth
    colour.reshape(-1, 3)[flat] = values

"""The receiver kernel, ported from default.shader.hlsl.

Every function here mirrors one in the shader and keeps its name, so the two
can be read against each other line by line. The port is vectorised over a flat
array of sample points rather than over a pixel quad: nothing in the shader's
shadow path reads across the quad (that is stated where ShadowLitLayers takes
its [loop]), so there is no derivative to lose.

Texture addressing is D3D11_TEXTURE_ADDRESS_BORDER with a border of 1.0, set in
gfx_direct3d11.cpp. A tap outside the slice therefore reads "far", which the
comparison comes out of as "lit" -- the same answer as not occluding.
"""

from __future__ import annotations

import numpy as np

from .capture import ShadowParams

DEPTH_MAX = 65535.0
GOLDEN_ANGLE = 2.3999632
TWO_PI = 6.28318530718


def _saturate(x):
    return np.clip(x, 0.0, 1.0)


def _smoothstep(lo, hi, x):
    t = _saturate((x - lo) / np.maximum(hi - lo, 1e-30))
    return t * t * (3.0 - 2.0 * t)


def project(world: np.ndarray, view_proj: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ShadowProject: world -> (uv, ndc z, inside).

    Row-vector against a row-major matrix, matching `mul(float4(p,1), viewProj)`
    with `row_major float4x4` storage.
    """
    homo = np.empty((world.shape[0], 4), np.float64)
    homo[:, :3] = world
    homo[:, 3] = 1.0
    clip = homo @ view_proj
    w = clip[:, 3]
    safe_w = np.where(np.abs(w) > 1e-6, w, 1e-6)
    ndc = clip[:, :3] / safe_w[:, None]
    uv = np.empty((world.shape[0], 2), np.float64)
    uv[:, 0] = ndc[:, 0] * 0.5 + 0.5
    uv[:, 1] = -ndc[:, 1] * 0.5 + 0.5
    inside = (
        (w > 0.0)
        & (np.abs(ndc[:, 0]) <= 1.0)
        & (np.abs(ndc[:, 1]) <= 1.0)
        & (ndc[:, 2] >= 0.0)
        & (ndc[:, 2] <= 1.0)
    )
    return uv, ndc[:, 2], inside


def _gather(slice_depth: np.ndarray, base: np.ndarray) -> np.ndarray:
    """The 2x2 footprint in Gather's component order: w=(0,0) z=(1,0) x=(0,1) y=(1,1).

    Out-of-range texels return 1.0, which is what the border sampler gives.
    """
    height, width = slice_depth.shape
    out = np.ones((base.shape[0], 4), np.float64)
    for comp, (dx, dy) in ((3, (0, 0)), (2, (1, 0)), (0, (0, 1)), (1, (1, 1))):
        tx = base[:, 0] + dx
        ty = base[:, 1] + dy
        ok = (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
        if ok.any():
            out[ok, comp] = slice_depth[ty[ok], tx[ok]] / DEPTH_MAX
    return out


def analytic_coverage(stored, z, sub, width) -> np.ndarray:
    """ShadowAnalyticCoverage: signed distance to the contour, in texels."""
    g = stored - z[:, None]
    row0 = g[:, 3] + (g[:, 2] - g[:, 3]) * sub[:, 0]
    row1 = g[:, 0] + (g[:, 1] - g[:, 0]) * sub[:, 0]
    value = row0 + (row1 - row0) * sub[:, 1]
    du = (g[:, 2] - g[:, 3]) + ((g[:, 1] - g[:, 0]) - (g[:, 2] - g[:, 3])) * sub[:, 1]
    dv = row1 - row0
    gradient = np.hypot(du, dv)
    flat = gradient < 1e-7
    out = _saturate(0.5 + (value / np.where(flat, 1.0, gradient)) / max(width, 1e-4))
    # step(0, value) where the quad holds no boundary to antialias.
    return np.where(flat, (value >= 0.0).astype(np.float64), out)


def jitter_noise(pixel: np.ndarray) -> np.ndarray:
    """ShadowJitterNoise: interleaved gradient noise of the pixel coordinate."""
    d = pixel[:, 0] * 0.06711056 + pixel[:, 1] * 0.00583715
    return np.modf(52.9829189 * np.modf(d)[0])[0]


def sample_pcf4(slice_depth, uv, z, texel_uv, p: ShadowParams) -> np.ndarray:
    """SampleShadowPCF4: compare four texels, then filter the comparisons."""
    texel_pos = uv / texel_uv - 0.5
    base = np.floor(texel_pos)
    sub = texel_pos - base
    stored = _gather(slice_depth, base.astype(np.int64))

    if p.edge[0] > 0.5:
        return analytic_coverage(stored, z, sub, float(p.edge[1]))

    # step(z, stored): 1 where the texel does not occlude.
    lit = (stored >= z[:, None]).astype(np.float64)
    inv = 1.0 - sub
    weights = np.stack(
        [
            inv[:, 0] * sub[:, 1],
            sub[:, 0] * sub[:, 1],
            sub[:, 0] * inv[:, 1],
            inv[:, 0] * inv[:, 1],
        ],
        axis=1,
    )
    return np.einsum("ij,ij->i", lit, weights)


def sample_jittered(slice_depth, uv, z, texel_uv, pixel, p: ShadowParams) -> np.ndarray:
    """SampleShadowJittered: a Vogel disk rotated per pixel."""
    if p.edge[2] < 0.5:
        return sample_pcf4(slice_depth, uv, z, texel_uv, p)

    taps = int(max(p.edge[3], 1.0))
    angle = jitter_noise(pixel) * TWO_PI
    rot = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    radius = float(p.jitter[0]) * texel_uv

    total = np.zeros(uv.shape[0], np.float64)
    for i in range(taps):
        r = np.sqrt((i + 0.5) / taps)
        theta = i * GOLDEN_ANGLE
        ux, uy = np.cos(theta), np.sin(theta)
        # Complex multiply: rotate the spiral's direction by this pixel's angle.
        dir_x = ux * rot[:, 0] - uy * rot[:, 1]
        dir_y = ux * rot[:, 1] + uy * rot[:, 0]
        offset = np.stack([dir_x, dir_y], axis=1) * (r * radius)
        total += sample_pcf4(slice_depth, uv + offset, z, texel_uv, p)
    return total / taps


def cascade_index(view_depth: np.ndarray, p: ShadowParams) -> np.ndarray:
    """ShadowCascadeIndex: first cascade whose far split still covers this depth."""
    count = p.cascade_count
    if count == 0:
        return np.zeros(view_depth.shape, np.int64)
    out = np.full(view_depth.shape, count - 1, np.int64)
    if count > 2:
        out = np.where(view_depth <= p.splits[2], 2, out)
    if count > 1:
        out = np.where(view_depth <= p.splits[1], 1, out)
    out = np.where(view_depth <= p.splits[0], 0, out)
    return np.minimum(out, count - 1)


def acne_slope(normal: np.ndarray, p: ShadowParams) -> np.ndarray:
    """ShadowAcneSlope: 1/N.L, capped so an edge-on surface stays bounded."""
    if p.acne1[0] < 0.5:
        return np.ones(normal.shape[0], np.float64)
    ndl = _saturate(normal @ -p.light_axis)
    return np.minimum(1.0 / np.maximum(ndl, 1.0e-3), p.acne1[1])


def acne_move_point(world, normal, texel_world, slope, p: ShadowParams) -> np.ndarray:
    """ShadowAcneMovePoint: offset ALONG THE SURFACE, not along the light."""
    if p.acne0[0] < 0.5:
        return world
    return world + normal * (texel_world * p.acne0[1] * slope)[:, None]


def harden_edge(coverage: np.ndarray, p: ShadowParams) -> np.ndarray:
    """ShadowHardenEdge: smoothstep the boundary, interiors map to themselves."""
    if p.harden[0] < 0.5:
        return coverage
    width = max((1.0 - p.harden[1]) * 0.5, 1.0e-5)
    threshold = float(p.harden[2])
    return _smoothstep(threshold - width, threshold + width, coverage)


def _actors_possible(world: np.ndarray, p: ShadowParams) -> np.ndarray:
    """The slab test from ShadowLitLayers.

    A receiver with none of the actor box behind it along the light cannot be
    shadowed by that layer. The box may be grown, never shrunk: a lookup landing
    where no actor was drawn reads the cleared depth and returns lit, which is
    exactly the value skipping leaves in place.
    """
    margin = float(np.max(p.texel_world[:3])) * 2.0
    box_lo = p.actor_min[:3] - margin
    box_hi = p.actor_max[:3] + margin
    if not np.all(box_lo <= box_hi):
        # An empty layer arrives inverted and stays inverted: no characters.
        return np.zeros(world.shape[0], bool)
    direction = -p.light_axis
    safe = np.where(np.abs(direction) < 1e-6, 1e-6, direction)
    t0 = (box_lo - world) / safe
    t1 = (box_hi - world) / safe
    t_near = np.minimum(t0, t1)
    t_far = np.maximum(t0, t1)
    t_enter = np.maximum(t_near.max(axis=1), -margin)
    return t_enter <= t_far.min(axis=1)


def lit_layers(
    world: np.ndarray,
    view_depth: np.ndarray,
    normal: np.ndarray,
    pixel: np.ndarray,
    world_slices: np.ndarray,
    actor_slices: np.ndarray | None,
    p: ShadowParams,
    want_actors: bool = True,
) -> np.ndarray:
    """ShadowLitLayers: (N, 2) of how lit the world and actor layers say each point is.

    The shader dispatches cascades on literal indices because ps_4_0 cannot index
    a vector by a runtime value; here the same shape falls out of grouping the
    points by cascade, which is also what lets one masked subset be sampled at a
    time instead of all three cascades for every point.
    """
    n = world.shape[0]
    lit = np.ones((n, 2), np.float64)
    count = p.cascade_count
    if count == 0:
        return lit

    cascade = cascade_index(view_depth, p)
    # Past the furthest point any cascade's footprint reaches there is nothing to
    # look up. The bound comes from the boxes, not the split ladder.
    in_reach = view_depth <= p.range_[0]
    if not in_reach.any():
        return lit

    acne_on = p.acne1[2] > 0.5
    slope = acne_slope(normal, p) if acne_on else np.zeros(n, np.float64)

    # Cross-fade band at the far edge of each cascade.
    far_edge = p.splits[np.minimum(cascade, 2)]
    near_edge = np.where(cascade == 0, 0.0, p.splits[np.maximum(cascade - 1, 0)])
    band_start = far_edge - (far_edge - near_edge) * p.params[1]
    blend = (cascade + 1 < count) & (view_depth > band_start) & in_reach
    t = np.where(blend, _smoothstep(band_start, far_edge, view_depth), 0.0)

    actors_ok = _actors_possible(world, p) if (want_actors and actor_slices is not None) else None
    actor_cascades = 0 if actor_slices is None else int(actor_slices.shape[0])

    for is_partner in (False, True):
        # The partner is the next coarser cascade, and gets its OWN acne offset:
        # the offset is sized in texels and the partner's texel is several times
        # larger, so sharing the primary's darkens the seam across the fade.
        target = np.minimum(cascade + 1, count - 1) if is_partner else cascade
        active_base = (blend & in_reach) if is_partner else in_reach

        for c in range(count):
            picked = active_base & (target == c)
            if not picked.any():
                continue
            idx = np.flatnonzero(picked)
            pos = world[idx]
            if acne_on:
                pos = acne_move_point(pos, normal[idx], float(p.texel_world[c]), slope[idx], p)

            uv, z, inside = project(pos, p.view_proj[c])
            # ShadowSample: outside the footprint there is nothing to occlude.
            # shadow_range.y == 1 inverts that to "occluded", which is debug view 1.
            default = 0.0 if abs(p.range_[1] - 1.0) < 0.5 else 1.0

            for layer in (0, 1):
                if layer == 1:
                    if not want_actors or actor_slices is None:
                        continue
                    if c >= actor_cascades:
                        # Past the actor layer's last cascade there is no slice.
                        # Faded to lit rather than skipped, so a character's
                        # shadow runs out instead of stopping at a line.
                        if is_partner:
                            lit[idx, 1] += (1.0 - lit[idx, 1]) * t[idx]
                        continue
                    keep = actors_ok[idx]
                    if not keep.any():
                        continue
                else:
                    keep = np.ones(idx.shape, bool)

                sub = np.flatnonzero(keep)
                depths = world_slices[c] if layer == 0 else actor_slices[c]
                texel_uv = float(p.texel_uv[c] if layer == 0 else p.actor_texel_uv[c])

                value = np.full(sub.shape[0], default, np.float64)
                vis = inside[sub]
                if vis.any():
                    hit = sub[vis]
                    value[vis] = sample_jittered(
                        depths, uv[hit], z[hit], texel_uv, pixel[idx[hit]], p
                    )
                rows = idx[sub]
                if is_partner:
                    lit[rows, layer] += (value - lit[rows, layer]) * t[rows]
                else:
                    lit[rows, layer] = value
    return lit


def shade(lit: np.ndarray, p: ShadowParams) -> tuple[np.ndarray, np.ndarray]:
    """PSMain's tail: combine the layers, harden, then apply darkness.

    Returns (coverage, visibility). Coverage is what debug view 5 shows -- the
    raw value before the hardening remap -- and visibility is the multiplier the
    lighting applies.
    """
    coverage = np.minimum(lit[:, 0], lit[:, 1])
    hardened = harden_edge(coverage, p)
    visibility = (1.0 - p.params[3]) + hardened * p.params[3]
    return coverage, visibility

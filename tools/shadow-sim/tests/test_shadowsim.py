"""Tests that need no capture and no GPU.

The fixtures below build a tiny synthetic scene -- an orthographic light looking
straight down at a 16x16 map with one raised block -- so the kernel's answers can
be reasoned about by hand. A test that can only be checked against a real capture
cannot say which of the two is wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shadowsim import kernel, sds  # noqa: E402
from shadowsim.capture import Capture, CaptureError  # noqa: E402
from shadowsim.reconstruct import plane, unproject  # noqa: E402

RES = 16
# World box the light covers: x,z in [-8, 8], y in [0, 32] with y up and the
# light looking straight down. Row-major, row-vector, mapping y -> ndc z in
# [0,1] with 1 furthest, which is what D3D and the capture use.
LIGHT = np.array(
    [
        [1 / 8, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1 / 32, 0.0],
        [0.0, 1 / 8, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]
)


def depth_of(y: float) -> int:
    """The u16 a surface at height y would be rasterised to."""
    return int(round((1.0 - y / 32.0) * 65535.0))


# The ground sits ABOVE the far plane on purpose. A surface at y=0 rasterises to
# 65535, which is the cleared value -- indistinguishable from "nothing was drawn
# here" -- and unproject drops it, correctly. Putting the fixture's floor there
# would test the reconstruction against a map it is right to call empty.
GROUND_Y = 2.0
BLOCK_Y = 10.0


def make_capture(tmp_path: Path, *, block=True, actors=True) -> Path:
    ground = np.full((1, RES, RES), depth_of(GROUND_Y), np.uint16)
    if block:
        # A block whose top is at BLOCK_Y, over one quadrant of the map.
        ground[0, :RES // 2, :RES // 2] = depth_of(BLOCK_Y)
    sds.save(tmp_path / "world.sds", ground)

    actor = np.full((1, RES, RES), 65535, np.uint16)
    if actors:
        actor[0, 10:12, 10:12] = depth_of(6.0)
    sds.save(tmp_path / "actors.sds", actor)

    meta = {
        "format": "SDS1", "complete": True, "layer": "world",
        "actor_layer": "actors.sds", "actor_slices": 1, "actor_own_texture": True,
        "slice_matrices": [LIGHT.ravel().tolist()],
        "slice_valid": [True],
        "shadow_view_proj": [LIGHT.ravel().tolist()],
        "shadow_splits": [1e9, 1e9, 1e9, 0.0],
        "shadow_texel_world": [1.0, 1.0, 1.0, 0.0],
        "shadow_texel_uv": [1 / RES, 1 / RES, 1 / RES, 0.0],
        "shadow_params": [1.0, 0.2, 0.0, 1.0],
        "shadow_range": [1e9, 0.0, 0.0, 0.0],
        "shadow_edge": [0.0, 2.0, 0.0, 8.0],
        "shadow_jitter": [2.0, 0.0, 0.0, 0.0],
        "shadow_acne0": [0.0, 0.0, 0.0, 0.0],
        "shadow_acne1": [0.0, 3.5, 0.0, 0.0],
        "shadow_harden": [0.0, 0.5, 0.5, 0.0],
        "shadow_actor_min": [-8.0, 0.0, -8.0, 0.0],
        "shadow_actor_max": [8.0, 32.0, 8.0, 0.0],
        "shadow_actor_texel_uv": [1 / RES, 1 / RES, 1 / RES, 0.0],
        "game_context": {"scene_id": 1},
    }
    (tmp_path / "capture.json").write_text(json.dumps(meta))
    return tmp_path


# --------------------------------------------------------------------- SDS1

def test_sds_round_trip(tmp_path):
    rng = np.random.default_rng(7)
    original = rng.integers(0, 65536, (2, 5, 9), dtype=np.uint16)
    # A long run as well as noise, so both branches of the encoder are covered.
    original[0, 0, :] = 4242
    sds.save(tmp_path / "r.sds", original)
    assert np.array_equal(sds.load(tmp_path / "r.sds"), original)


def test_sds_rejects_foreign_file(tmp_path):
    (tmp_path / "bad.sds").write_bytes(b"NOPE" + bytes(12))
    with pytest.raises(sds.SdsError, match="magic"):
        sds.load(tmp_path / "bad.sds")


def test_sds_rejects_trailing_bytes(tmp_path):
    sds.save(tmp_path / "t.sds", np.zeros((1, 2, 2), np.uint16))
    with open(tmp_path / "t.sds", "ab") as handle:
        handle.write(b"\0")
    with pytest.raises(sds.SdsError, match="trailing"):
        sds.load(tmp_path / "t.sds")


# ------------------------------------------------------------------ capture

def test_capture_reports_a_missing_field(tmp_path):
    make_capture(tmp_path)
    meta = json.loads((tmp_path / "capture.json").read_text())
    del meta["shadow_acne0"]
    (tmp_path / "capture.json").write_text(json.dumps(meta))
    with pytest.raises(CaptureError, match="shadow_acne0"):
        Capture.load(tmp_path)


def test_capture_rejects_an_interrupted_capture(tmp_path):
    make_capture(tmp_path)
    meta = json.loads((tmp_path / "capture.json").read_text())
    meta["complete"] = False
    (tmp_path / "capture.json").write_text(json.dumps(meta))
    with pytest.raises(CaptureError, match="complete"):
        Capture.load(tmp_path)


# ----------------------------------------------------------------- geometry

def test_projection_round_trips_through_the_matrix(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    points = np.array([[0.0, 0.0, 0.0], [3.5, 8.0, -2.25], [-7.9, 31.0, 7.9]])
    uv, z, inside = kernel.project(points, capture.params.view_proj[0])
    assert inside.all()
    # uv and z back to world, the way reconstruct.unproject does it.
    ndc = np.stack([uv[:, 0] * 2 - 1, 1 - uv[:, 1] * 2, z, np.ones(len(z))], axis=1)
    back = ndc @ np.linalg.inv(capture.params.view_proj[0])
    assert np.allclose(back[:, :3] / back[:, 3:4], points, atol=1e-9)


def test_unproject_recovers_the_block_height(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    surface = unproject(capture, 0)
    heights = surface.world[:, 1]
    assert np.isclose(heights.max(), BLOCK_Y, atol=1e-3)
    assert np.isclose(heights.min(), GROUND_Y, atol=1e-3)
    # A quarter of the map is the raised block.
    midpoint = (GROUND_Y + BLOCK_Y) * 0.5
    assert np.isclose((heights > midpoint).mean(), 0.25, atol=0.02)


def test_reconstructed_normals_face_the_light(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    surface = unproject(capture, 0)
    assert (surface.normal @ -capture.params.light_axis >= 0.0).all()


# ------------------------------------------------------------------- kernel

def test_a_point_on_the_stored_surface_is_lit(tmp_path):
    """step(z, stored) is stored >= z, so equal depths do not occlude."""
    capture = Capture.load(make_capture(tmp_path))
    surface = unproject(capture, 0)
    lit = kernel.lit_layers(
        surface.world, np.zeros(len(surface.world)), surface.normal,
        surface.texel.astype(float), capture.world, None, capture.params,
        want_actors=False,
    )
    assert lit[:, 0].min() > 0.999


def test_ground_under_the_block_is_occluded(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    # Straight under the block, on the ground the block stands on.
    # uv.x = x/16 + 0.5 puts column 0 at x = -8; uv.y = -z/16 + 0.5 puts ROW 0
    # at z = +8. The quadrant the block covers is therefore x < 0 and z > 0, and
    # getting that backwards is why this test is written against both points.
    # A HAIR above the floor, not on it. A receiver at exactly the stored height
    # compares against a depth that 16-bit quantisation rounded the wrong way and
    # self-shadows -- that is acne, and it has its own tests below. Here the
    # question is only whether the block occludes, so the ambiguity is stepped over.
    under = np.array([[-4.0, GROUND_Y + 0.1, 4.0]])
    clear = np.array([[4.0, GROUND_Y + 0.1, -4.0]])
    for point, expected in ((under, 0.0), (clear, 1.0)):
        normal = np.array([[0.0, 1.0, 0.0]])
        lit = kernel.lit_layers(
            point, np.zeros(1), normal, np.zeros((1, 2)),
            capture.world, None, capture.params, want_actors=False,
        )
        assert lit[0, 0] == pytest.approx(expected)


def test_the_actor_layer_casts_its_own_shadow(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    # Ground under the 2x2 actor patch at texels (10..11, 10..11): its centre is
    # x = -8 + 10.5 = 2.5 and z = 8 - 10.5 = -2.5.
    point = np.array([[2.5, GROUND_Y + 0.1, -2.5]])
    normal = np.array([[0.0, 1.0, 0.0]])
    lit = kernel.lit_layers(
        point, np.zeros(1), normal, np.zeros((1, 2)),
        capture.world, capture.actors, capture.params, want_actors=True,
    )
    assert lit[0, 1] < 0.5, "the actor layer should occlude here"
    lit_without = kernel.lit_layers(
        point, np.zeros(1), normal, np.zeros((1, 2)),
        capture.world, capture.actors, capture.params, want_actors=False,
    )
    assert lit_without[0, 1] == 1.0, "want_actors=False must skip the layer"


def test_outside_the_footprint_is_lit_not_dark(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    far = np.array([[1000.0, 0.0, 1000.0]])
    lit = kernel.lit_layers(
        far, np.zeros(1), np.array([[0.0, 1.0, 0.0]]), np.zeros((1, 2)),
        capture.world, None, capture.params, want_actors=False,
    )
    assert lit[0, 0] == 1.0


def test_debug_view_one_inverts_the_rejected_case(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    capture.params.range_[1] = 1.0  # shadow_range.y == 1
    far = np.array([[1000.0, 0.0, 1000.0]])
    lit = kernel.lit_layers(
        far, np.zeros(1), np.array([[0.0, 1.0, 0.0]]), np.zeros((1, 2)),
        capture.world, None, capture.params, want_actors=False,
    )
    assert lit[0, 0] == 0.0


def test_acne_offset_moves_along_the_surface_normal(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    p = capture.params
    p.acne0[0], p.acne0[1] = 1.0, 2.0
    world = np.array([[0.0, 0.0, 0.0]])
    normal = np.array([[0.0, 1.0, 0.0]])
    moved = kernel.acne_move_point(world, normal, 1.0, np.ones(1), p)
    assert np.allclose(moved, [[0.0, 2.0, 0.0]])
    p.acne0[0] = 0.0
    assert np.allclose(kernel.acne_move_point(world, normal, 1.0, np.ones(1), p), world)


def test_harden_leaves_interiors_alone(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    p = capture.params
    p.harden[0], p.harden[1], p.harden[2] = 1.0, 0.5, 0.5
    out = kernel.harden_edge(np.array([0.0, 0.5, 1.0]), p)
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)


def test_jitter_noise_stays_in_range():
    pixels = np.stack(np.meshgrid(np.arange(64), np.arange(64)), -1).reshape(-1, 2)
    noise = kernel.jitter_noise(pixels.astype(float))
    assert noise.min() >= 0.0 and noise.max() < 1.0
    # A constant would also satisfy the bounds; this is what says it varies.
    assert noise.std() > 0.2


def test_cascade_index_follows_the_split_ladder(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    p = capture.params
    p.params[0] = 3.0
    p.splits[:3] = [350.0, 2500.0, 6000.0]
    depths = np.array([0.0, 349.0, 351.0, 2499.0, 2501.0, 5999.0, 99999.0])
    assert kernel.cascade_index(depths, p).tolist() == [0, 0, 1, 1, 2, 2, 2]


def _plane_occlusion(capture, height_y) -> float:
    surface = plane(capture, height_y=height_y, centre=np.zeros(3), extent=15.0, resolution=32)
    lit = kernel.lit_layers(
        surface.world, np.zeros(len(surface.world)), surface.normal,
        surface.texel.astype(float), capture.world, None, capture.params,
        want_actors=False,
    )
    return float((lit[:, 0] < 0.5).mean())


def test_plane_receiver_takes_the_block_shadow(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    # Clear of the quantisation, so what is measured is the block's shadow.
    occluded = _plane_occlusion(capture, GROUND_Y + 0.1)
    # The block covers a quarter of the map and the light is straight down.
    assert 0.2 < occluded < 0.3


def test_a_receiver_on_the_stored_surface_self_shadows(tmp_path):
    """Shadow acne, reproduced rather than modelled.

    depth_of(2.0) rounds 0.9375 * 65535 to 61439, and 61439 / 65535 is very
    slightly BELOW 0.9375. The receiver at that exact height therefore compares
    as further than the depth its own rasterisation wrote, and the whole plane
    comes back occluded. This is the artefact the acne controls exist for, and a
    capture reproduces it because it carries the quantised depths themselves.
    """
    capture = Capture.load(make_capture(tmp_path))
    assert _plane_occlusion(capture, GROUND_Y) > 0.99


def test_the_acne_offset_clears_the_self_shadow(tmp_path):
    capture = Capture.load(make_capture(tmp_path))
    p = capture.params
    p.acne1[2] = 1.0  # master
    p.acne0[0], p.acne0[1] = 1.0, 0.6  # move along the normal, 0.6 texels
    occluded = _plane_occlusion(capture, GROUND_Y)
    assert 0.2 < occluded < 0.3, "only the block should be left occluding"

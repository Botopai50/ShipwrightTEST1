"""Command line over the simulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .capture import Capture, CaptureError
from .render import (
    default_camera, default_eye, receivers, render_camera_view,
    render_light_view, render_plane_view, render_scene_view, to_image,
)

CHANNELS = ("visibility", "coverage", "layers", "cascade", "normal", "depth")

# The float4 lanes a sweep may address, by the name the shader gives them.
FIELDS = {
    "splits": "splits", "texel_world": "texel_world", "texel_uv": "texel_uv",
    "params": "params", "range": "range_", "edge": "edge", "jitter": "jitter",
    "acne0": "acne0", "acne1": "acne1", "harden": "harden",
}
LANES = {"x": 0, "y": 1, "z": 2, "w": 3}


def apply_override(params, spec: str) -> None:
    """`--set acne0.y=3` writes shadow_acne0.y, the acne offset in texels."""
    name, _, value = spec.partition("=")
    field, _, lane = name.strip().partition(".")
    if field not in FIELDS or lane not in LANES or not value:
        raise SystemExit(
            f"--set {spec!r}: expected <field>.<xyzw>=<number> where field is "
            f"one of {', '.join(sorted(FIELDS))}"
        )
    try:
        number = float(value)
    except ValueError:
        raise SystemExit(f"--set {spec!r}: {value!r} is not a number") from None
    getattr(params, FIELDS[field])[LANES[lane]] = number


def save(path: Path, image: np.ndarray) -> None:
    from PIL import Image

    data = np.clip(image, 0.0, 1.0)
    array = (data * 255.0 + 0.5).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array if array.ndim == 3 else array, "RGB" if array.ndim == 3 else "L").save(path)
    print(f"  wrote {path}")


def parse_vec3(text: str, what: str) -> np.ndarray:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise SystemExit(f"{what}: expected three numbers, got {text!r}")
    return np.array([float(p) for p in parts], np.float64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="shadowsim",
        description="Reproduce a Fast3D shadow capture offline.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("capture", type=Path, help="directory holding capture.json")
    common.add_argument("--set", action="append", default=[], metavar="FIELD.LANE=N",
                        help="override one kernel input, e.g. --set acne0.y=3")
    common.add_argument("--no-actors", action="store_true", help="skip the actor layer")
    common.add_argument("--step", type=int, default=2,
                        help="sample every Nth texel (default 2; 1 is every texel)")
    common.add_argument("-o", "--out", type=Path, default=Path("shadowsim-out"))
    common.add_argument("--channel", action="append", default=[], choices=CHANNELS,
                        help="what to draw; repeatable (default visibility and coverage)")

    ground = argparse.ArgumentParser(add_help=False)
    ground.add_argument("--plane-y", type=float, default=0.0,
                        help="height of the flat receiver, world units (default 0)")
    ground.add_argument("--extent", type=float, default=6000.0,
                        help="side of the flat receiver, world units (default 6000)")
    ground.add_argument("--resolution", type=int, default=512,
                        help="samples per side of the flat receiver (default 512)")

    p_info = sub.add_parser("info", help="describe a capture")
    p_info.add_argument("capture", type=Path)

    p_light = sub.add_parser("light", parents=[common], help="render in the light's own grid")
    p_light.add_argument("--cascade", type=int, default=0)
    p_light.add_argument("--eye", type=str, default=None,
                         help="camera position 'x,y,z'; decides the cascade ladder")

    p_scene = sub.add_parser("scene", parents=[common],
                             help="shade the camera's own depth buffer: the frame as rendered")
    p_scene.add_argument("--caster-shift", type=float, default=0.0, metavar="N",
                         help="move each sample N world units along the light before "
                              "projecting; a measuring instrument, not a shader input")

    p_plane = sub.add_parser("plane", parents=[common, ground],
                             help="shade a flat receiver: the cast shadow itself")
    p_plane.add_argument("--eye", type=str, default=None)

    p_cam = sub.add_parser("camera", parents=[common, ground], help="render from a viewpoint")
    p_cam.add_argument("--receiver", choices=("surface", "plane", "both"), default="both")
    p_cam.add_argument("--cascade", action="append", type=int, default=[],
                       help="which cascades to reconstruct; repeatable (default all)")
    p_cam.add_argument("--eye", type=str, default=None)
    p_cam.add_argument("--target", type=str, default=None)
    p_cam.add_argument("--width", type=int, default=960)
    p_cam.add_argument("--height", type=int, default=540)
    p_cam.add_argument("--fov", type=float, default=60.0)
    p_cam.add_argument("--splat", type=int, default=2)

    args = ap.parse_args(argv)
    try:
        capture = Capture.load(args.capture)
    except CaptureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "info":
        print(capture.describe())
        return 0

    for spec in args.set:
        apply_override(capture.params, spec)
    if args.set:
        print("overrides applied:", ", ".join(args.set))

    channels = args.channel or ["visibility", "coverage"]
    want_actors = not args.no_actors

    if args.command == "light":
        eye = default_eye(capture) if args.eye is None else parse_vec3(args.eye, "--eye")
        print(f"eye {np.array2string(eye, precision=1)}  cascade {args.cascade}  step {args.step}")
        shaded = render_light_view(capture, args.cascade, args.step, eye, want_actors)
        occluded = float((shaded.coverage < 0.5).mean())
        print(f"  {len(shaded.coverage)} surface samples, {occluded:.1%} more than half occluded")
        for channel in channels:
            save(args.out / f"light-c{args.cascade}-{channel}.png", to_image(shaded, channel))
        return 0

    if args.command == "scene":
        if capture.scene is None:
            print("error: this capture carries no receiver -- "
                  f"{capture.meta.get('camera_note') or 'taken by a build that wrote none'}",
                  file=sys.stderr)
            return 2
        shaded = render_scene_view(capture, args.step, want_actors, args.caster_shift)
        occluded = float((shaded.coverage < 0.5).mean())
        print(f"  {len(shaded.coverage)} visible surface samples,"
              f" {occluded:.1%} more than half occluded"
              + (f"  (caster shift {args.caster_shift:+g})" if args.caster_shift else ""))
        tag = f"{args.caster_shift:+g}" if args.caster_shift else "asis"
        for channel in channels:
            save(args.out / f"scene-{tag}-{channel}.png", to_image(shaded, channel))
        return 0

    if args.command == "plane":
        eye = default_eye(capture) if args.eye is None else parse_vec3(args.eye, "--eye")
        print(f"eye {np.array2string(eye, precision=1)}  plane y={args.plane_y:g}"
              f"  extent {args.extent:g}  {args.resolution}x{args.resolution}")
        shaded = render_plane_view(capture, args.plane_y, args.extent, args.resolution,
                                   None, eye, want_actors)
        occluded = float((shaded.coverage < 0.5).mean())
        print(f"  {occluded:.1%} of the receiver is more than half occluded")
        for channel in channels:
            save(args.out / f"plane-{channel}.png", to_image(shaded, channel))
        return 0

    centre = default_eye(capture)
    target = centre if args.target is None else parse_vec3(args.target, "--target")
    if args.eye is not None:
        eye = parse_vec3(args.eye, "--eye")
    else:
        eye = default_camera(capture, target)
    print(f"eye {np.array2string(eye, precision=1)} -> target {np.array2string(target, precision=1)}")
    cascades = tuple(args.cascade) if args.cascade else (0, 1, 2)
    surfaces = receivers(capture, args.receiver, args.step, cascades,
                         args.plane_y, args.extent, args.resolution)
    print(f"  receiver {args.receiver}: {len(surfaces)} surfaces,"
          f" {sum(len(s.world) for s in surfaces)} samples")
    for channel in channels:
        image = render_camera_view(
            capture, eye, target, args.width, args.height, args.fov,
            surfaces=surfaces, channel=channel, splat=args.splat, want_actors=want_actors,
        )
        save(args.out / f"camera-{channel}.png", image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

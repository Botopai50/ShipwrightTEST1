# shadowsim — reproduce a shadow capture offline

Loads a `shadow-captures/<stamp>` directory written by the in-game **Salvar
captura das sombras (DirectX)** button and re-runs the receiver kernel over it on
the CPU, with no game, no GPU and no ROM. The point is to be able to change one
input and look at the result — which is what the button cannot do, because by the
time a capture exists the frame that produced it is gone.

```sh
pip install -r requirements.txt
python -m shadowsim info    shadow-captures/1789385961619299
python -m shadowsim plane   shadow-captures/1789385961619299 -o out
python -m shadowsim camera  shadow-captures/1789385961619299 -o out
python -m shadowsim light   shadow-captures/1789385961619299 --cascade 0 -o out
```

## What a capture contains, and what follows from it

A capture holds the depth slices of both caster layers and every `float4` the
receiver reads — the whole of `PerShadowCB`. It does **not** hold the camera, the
colour buffer, or the geometry the shadow lands on.

That last omission is the one that shapes this tool. A shadow map stores the
surface **nearest the light**, so unprojecting it gives back the lit surface and
nothing else: the ground lying in the castle's shadow is behind the castle along
the light and was never written. Shade that reconstruction against its own map
and almost every point comes back lit — correctly, and uselessly.

So there are two receivers, and the choice between them is the first thing to
make when reading a result.

| Receiver | What it is | What it answers |
| --- | --- | --- |
| `plane` | A flat grid at a chosen height | **The cast shadow.** What the castle throws on the ground |
| `surface` | The light's own view, unprojected | **Self-shadowing.** Acne, bias, the edge kernel, cascade seams |

Both are real receivers: each point is projected into the captured maps and run
through the captured kernel. The plane is idealised ground rather than the
scene's own — exact shadow, invented floor.

## Views

* `plane` — shades the flat receiver in its own grid. Start here.
* `light` — one sample per shadow-map texel, nothing resampled. Every pixel is a
  real surface point, so this is the view for the comparison itself.
* `camera` — a viewpoint. The ground is intersected one ray per pixel (dense and
  exact); the reconstructed geometry is point-splatted over it with a depth test,
  so it thins out close to the eye. Shapes and boundaries are faithful; a
  per-pixel match against a screenshot is not what it is for.

Channels mirror the shader's own debug selectors (`shadow_range.y`):
`visibility`, `coverage` (mode 5), `layers` (mode 2), `cascade` (mode 7),
`normal` (mode 3), `depth`.

## Changing an input

`--set FIELD.LANE=N` writes one lane of one `float4` before anything is shaded,
using the shader's names:

```sh
# the acne offset, in texels — capture above was taken at 0.6
python -m shadowsim plane CAP --set acne0.y=0 -o out-no-bias
python -m shadowsim plane CAP --set acne0.y=3 -o out-heavy-bias

# turn the analytic edge on, and widen its ramp
python -m shadowsim plane CAP --set edge.x=1 --set edge.y=4

# jitter on, 16 taps, 3-texel disk
python -m shadowsim plane CAP --set edge.z=1 --set edge.w=16 --set jitter.x=3
```

Fields: `splits`, `texel_world`, `texel_uv`, `params`, `range`, `edge`, `jitter`,
`acne0`, `acne1`, `harden`. Lanes are `x`/`y`/`z`/`w`. Read
`fast/shadow_map.h` and the `PerShadowCB` comments for what each lane means.

## The camera, and why it matters

The capture carries none, so `--eye` is a choice. It is not cosmetic: **the
cascade a point lands in is chosen from its distance to the camera**, so the eye
decides which resolution, which texel size and which cross-fade every sample
gets. The default is the centre of the actor box — where the characters were, so
within a few units of what the player was looking at.

Negative coordinates need `=`, or `argparse` reads the leading `-` as a flag:

```sh
python -m shadowsim camera CAP --eye=-1600,1500,3400 --target=800,0,1900
```

## Fidelity

`kernel.py` is a port of the shadow path in
`libultraship/src/fast/shaders/directx/default.shader.hlsl`, function by function
and under the same names, so the two can be read side by side. It reproduces
projection and the `inside` test, the 2×2 comparison and its bilinear weights,
the analytic coverage path, the Vogel-spiral jitter and its per-pixel rotation,
the cascade ladder and cross-fade, both acne corrections, the actor-layer slab
test, the hardening remap, and the `BORDER` addressing with a white border.

It is a port, not the shader. Two differences are known and neither is small
enough to leave implied: it runs in float64 where the GPU runs float32, and it
samples texels directly where the hardware resolves a footprint in fixed point.
Neither changes what any of the controls does; both mean a value here can differ
from the GPU's in its last bits.

## Tests

```sh
python -m pytest tests/
```

Twenty tests, no capture and no GPU needed: they build a 16×16 map with one
raised block and check the kernel against answers that can be worked out by hand.
Two of them pin the acne behaviour — a receiver at exactly the height its own
rasterisation wrote comes back fully occluded, because 16-bit quantisation rounds
the stored depth just below it, and the 0.6-texel offset clears it.

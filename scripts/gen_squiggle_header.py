#!/usr/bin/env python3
"""Generate docs/images/squiggle-header.png — the nanopore-squiggle watermark
drawn behind the site header/footer bar (docs/stylesheets/space.css).

The original PNG (#123/#124) was committed with no generator. This script
reproduces its visual contract, reverse-engineered from the shipped pixels:
a random staircase of levels connected by vertical transitions (the way a
segmented nanopore current trace is normally drawn), solid mid-grey
(128, 128, 128) at a max alpha of 89/255 (~35%), anti-aliased, on a
transparent background, sized 6400x150 so it can be scaled to any header
width by the CSS `background-size: 100% 100%`.

Needs numpy (already a transitive dependency via pandas in the default pixi
env) but nothing else — the PNG is encoded by hand with stdlib zlib/struct so
the docs toolchain does not need Pillow/matplotlib.

Usage:
    pixi run python scripts/gen_squiggle_header.py
    pixi run python scripts/gen_squiggle_header.py --seed 7 --out /tmp/preview.png
"""

from __future__ import annotations

import argparse
import struct
import zlib

import numpy as np


def build_trace(
    width: int,
    height: int,
    seed: int,
    supersample: int,
    line_width: float,
    min_segment: int,
    max_segment: int,
    jitter: float,
    margin_top: float,
    margin_bottom: float,
) -> np.ndarray:
    """Return an (height, width) float32 array of alpha coverage in [0, 1]."""
    rng = np.random.default_rng(seed)

    sx = width * supersample
    sy = height * supersample
    y_top = margin_top * sy
    y_bot = (1.0 - margin_bottom) * sy
    band = y_bot - y_top

    # Random segment durations (in supersampled px) covering the full width.
    lengths = []
    total = 0
    lo, hi = min_segment * supersample, max_segment * supersample
    while total < sx:
        seg = int(rng.integers(lo, hi + 1))
        lengths.append(seg)
        total += seg
    lengths[-1] -= total - sx  # trim the last segment to land exactly on sx

    # Mostly-independent levels per segment (light AR(1) correlation so
    # consecutive levels aren't *pure* noise), normalized to [0, 1]. A real
    # segmented current trace jumps between largely unrelated levels rather
    # than drifting smoothly, which is what gives it a jagged skyline look
    # instead of rolling hills.
    n = len(lengths)
    level = 0.5
    levels = np.empty(n, dtype=np.float64)
    for i in range(n):
        level = 0.25 * level + 0.75 * float(rng.normal(0.5, 0.30))
        level = float(np.clip(level, 0.03, 0.97))
        levels[i] = level

    # Per-sample y trace: flat within a segment plus small jitter, i.e. a
    # staircase — the vertical jump between segments is what gives the
    # rasterizer below its tall "transition" columns.
    y = np.empty(sx + 1, dtype=np.float64)
    pos = 0
    for seg_len, lvl in zip(lengths, levels):
        y_val = y_bot - lvl * band
        y[pos : pos + seg_len] = y_val
        pos += seg_len
    y[sx] = y[sx - 1]
    y += rng.normal(0.0, jitter * band, size=sx + 1)
    y = np.clip(y, 0, sy - 1)

    # Rasterize as a polyline: column x covers [min(y[x], y[x+1]), max(...)]
    # thickened by the stroke half-width, exactly like a line renderer would.
    half_w = (line_width * supersample) / 2.0
    y0, y1 = y[:-1], y[1:]
    top = np.minimum(y0, y1) - half_w
    bot = np.maximum(y0, y1) + half_w

    rows = np.arange(sy, dtype=np.float64)[:, None]
    mask = (rows >= top[None, :]) & (rows <= bot[None, :])

    # Box-filter downsample by `supersample` in both axes -> anti-aliasing.
    coverage = mask.reshape(height, supersample, width, supersample).mean(axis=(1, 3))
    return coverage.astype(np.float32)


def write_png(path: str, rgba: np.ndarray) -> None:
    """Write an (H, W, 4) uint8 array as an 8-bit RGBA PNG, stdlib only."""
    height, width, _ = rgba.shape
    raw = bytearray()
    for row in rgba:
        raw.append(0)  # filter type 0 (none) per scanline
        raw.extend(row.tobytes())

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(bytes(raw), level=9)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", idat))
        f.write(chunk(b"IEND", b""))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="docs/images/squiggle-header.png")
    p.add_argument("--width", type=int, default=6400)
    p.add_argument("--height", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--supersample", type=int, default=4, help="AA supersampling factor")
    p.add_argument("--color", default="128,128,128", help="R,G,B (0-255)")
    p.add_argument(
        "--max-alpha",
        type=int,
        default=89,
        help="peak alpha, 0-255 (89 matches the original)",
    )
    p.add_argument(
        "--line-width", type=float, default=1.6, help="stroke width in final px"
    )
    p.add_argument(
        "--min-segment", type=int, default=8, help="min plateau width in final px"
    )
    p.add_argument(
        "--max-segment", type=int, default=70, help="max plateau width in final px"
    )
    p.add_argument(
        "--jitter",
        type=float,
        default=0.025,
        help="per-sample noise, fraction of vertical band",
    )
    p.add_argument(
        "--margin-top",
        type=float,
        default=0.15,
        help="fraction of height kept blank at top",
    )
    p.add_argument(
        "--margin-bottom",
        type=float,
        default=0.15,
        help="fraction of height kept blank at bottom",
    )
    args = p.parse_args()

    coverage = build_trace(
        args.width,
        args.height,
        args.seed,
        args.supersample,
        args.line_width,
        args.min_segment,
        args.max_segment,
        args.jitter,
        args.margin_top,
        args.margin_bottom,
    )

    r, g, b = (int(v) for v in args.color.split(","))
    alpha = np.clip(coverage * args.max_alpha, 0, 255).astype(np.uint8)
    rgba = np.zeros((args.height, args.width, 4), dtype=np.uint8)
    rgba[..., 0] = r
    rgba[..., 1] = g
    rgba[..., 2] = b
    rgba[..., 3] = alpha

    write_png(args.out, rgba)
    print(f"wrote {args.out} ({args.width}x{args.height}, seed={args.seed})")


if __name__ == "__main__":
    main()

# Image assets

Most files here are the animated CLI demos — see `docs/tapes/README.md` for
how those are regenerated.

## `squiggle-header.png`

The nanopore-squiggle watermark drawn behind the site header/footer bar
(`docs/stylesheets/space.css`, `.md-header::before` / `.md-footer-meta::before`).
Regenerate it with:

```bash
pixi run python scripts/gen_squiggle_header.py
```

That reproduces the shipped asset's contract — 6400x150, mid-grey
(128, 128, 128) at up to 35% alpha, transparent background — from a
synthetic random staircase rather than a fixed image, so a different `--seed`
gives a different (but equally plausible) trace. See the script's `--help`
for the generation knobs (segment length, jitter, line width, ...).

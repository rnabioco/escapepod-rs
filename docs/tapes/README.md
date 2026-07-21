# Terminal-demo tapes

The `*.tape` files here are [VHS](https://github.com/charmbracelet/vhs) scripts
that render the animated CLI demos committed under `docs/images/` (referenced from
the per-command docs pages and the top-level `README.md`).

## Regenerate on a local macOS laptop only

VHS does **not** render terminals in a lightweight way: it runs each tape's
commands inside a `ttyd` server, loads that page in a **headless Chromium** (via
go-rod), screenshots every frame, and pipes them to `ffmpeg`. That Chromium +
ffmpeg workload will overwhelm the 2-core cluster login node and must never run on
a compute node. Regeneration is therefore a **macOS-local** workflow, provisioned
with Homebrew rather than pixi/conda — there is no cluster path.

### One-time setup

```bash
brew install vhs                    # also installs ttyd + ffmpeg
brew install --cask font-fira-code  # the tapes pin `Set FontFamily "Fira Code"`
```

### Regenerate

From the repository root:

```bash
./docs/tapes/generate.sh
```

The script guards on macOS (it exits immediately elsewhere), checks the
prerequisites, builds the release `escpod` binary so the gifs show current CLI
output, and renders every `docs/tapes/*.tape` to `docs/images/`.

### Sample data

The tapes read sample POD5 files under `data/drna/` (gitignored), so that data
must be present locally for the gifs to render.

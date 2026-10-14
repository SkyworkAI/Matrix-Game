# Validation

Run from the source root after installing the corresponding dependencies.
The full Python suite requires torch and torchvision even when running entirely
on CPU; it does not require model weights or a GPU. A lightweight browser/media
installation alone is insufficient for the tensor and preprocessing tests.

## CPU and Native

```bash
python -m unittest discover -s tests -p 'test_*.py'
node tests/browser/test_audio.cjs
node tests/browser/test_hud.cjs
bash scripts/build_game.sh
```

The Python suite includes CLI dispatch, pixel-exact preprocessing against the
training torchvision/PIL path, last-block padding, sink/cache contracts, early
publication, bounded input admission, UI pairing, HTTP boundaries and resource
path/hash validation. Model tensors are mocked in the stream contract tests;
these tests are not a GPU rendering-quality evaluation.

With the asset bundle installed:

```bash
MATRIX_GAME_DATA_DIR="$PWD/resources/game" game/bin/MedievalCastleBBoxMap --scene castle --validate
MATRIX_GAME_DATA_DIR="$PWD/resources/game" game/bin/MedievalCastleBBoxMap --scene ancient-egypt --validate
MATRIX_GAME_DATA_DIR="$PWD/resources/game" game/bin/MedievalCastleBBoxMap --scene castle --scene-test
MATRIX_GAME_DATA_DIR="$PWD/resources/game" game/bin/MedievalCastleBBoxMap --scene ancient-egypt --map-surface-test
```

Scene visual checks require a usable OpenGL display/context. Their generated
reports/captures are not release assets.

## Browser Audio

Install Playwright outside the source tree, including Chromium and WebKit.
`PLAYWRIGHT_MODULE` can point to its installed package. `CHROME_PATH` optionally
selects a system Chrome. The browser test starts and closes a local test server:

```bash
node tests/browser/test_audio_browser.cjs resources/game
```

Test a live session in Safari/Chrome too. Synthetic browser audio checks do not
measure network latency or validate RTP/HUD alignment end-to-end.

## Browser Integration

Against an isolated running service (no other active player):

```bash
TEST_URL=http://127.0.0.1:8025 TEST_SCENE=ancient-egypt \
  node tests/browser/test_browser.cjs ../test-results/browser
TEST_URL=http://127.0.0.1:8025 STABILITY_SECONDS=180 \
  node tests/browser/test_stability.cjs ../test-results/stability
```

The first test exercises reference/model/dual views, frame-matched HUD and audio,
mouse input, responsive layouts, Stop and session restart. The second checks
continuous progress and bounded client/media state, then disconnect cleanup and
reconnection. Set `TEST_WEBKIT=1` for Playwright WebKit. If headless WebKit does not
provide pointer lock, use `DISPLAY_ONLY=1` for display/metadata/HUD checks only;
the report explicitly records that input checks were skipped. This is not a
substitute for a manual Safari check. Test reports contain runtime diagnostics and must not
be distributed as source. Long tests may legitimately end if the player dies;
that is distinct from a transport or inference stall.

For the racing server:

```bash
bash scripts/build_racing.sh -DRACING_ASSET_ROOT="$PWD/resources/racing"
python -m unittest discover -s game/racing/tools -p 'test_*.py'
TEST_URL=http://127.0.0.1:8025 TEST_MODE=model TEST_SECONDS=90 \
  RACE_MODES=time_trial TEST_SHUTTLE=1 EXPECT_DYNAMIC=1 \
  node tests/browser/test_racing.cjs ../test-results/racing-dynamic
```

Use `--prompt-mode dynamic --prompt-interval-chunks 1` on the server for this
test. Omit `EXPECT_DYNAMIC` for fixed mode. `RACE_MODES=time_trial,pursuit` also
checks both modes and session restart; `TEST_MODE=reference` omits model work.
The test reports video-pixel variance, desktop/mobile screenshots, bounded
queues, live frame progress and actual prompt updates. Reverse/forward controls
exercise changing captions; this is not an autonomous route-completion test.

## GPU Release Gate

Render the same recorded clean RGB with the same seed, prompt and sink using
`--dit-runtime native --no-vae-graph`, then the fast runtime with the same codecs
and timesteps. Include more than 21 latent chunks to exercise cache eviction.
Compare decoded clips and numerical differences; ring/fused is not expected to
be bitwise-equivalent. Test Wan/two-step and TAEHV/one-step separately, without
confounding codec, scheduler or prompt changes with runtime comparisons.

Use `--profile-chunks` online for bounded per-stage reports. Measure warmup
separately. A GPU throughput number is not an action-to-display latency number.

## Distribution

```bash
python scripts/package_release.py source --output ../matrix-game-4.0-source-release
python scripts/package_release.py resources --root ../matrix-game-4.0-resources \
  --manifest releases/game-assets.json --output ../matrix-game-4.0-assets-release
```

Outputs must be new directories. Source packaging excludes generated files,
rejects source symlinks and unexpected model/media files, scans known private
deployment identifiers, and emits SHA256s. Resource packaging uses only the
manifest allowlist. This automated check complements, rather than replaces,
a manual secret/license review. Neither command uploads anything.

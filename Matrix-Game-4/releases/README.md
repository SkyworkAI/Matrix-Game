# Resource Distributions

Model checkpoints, sink images, the fast codec and game assets are published at
[Skywork/Matrix-Game-4.0](https://huggingface.co/Skywork/Matrix-Game-4.0).
Both manifests use this **model** repository, not a separate dataset repository.

Pinned release revision: `4419accdb7bb7881234c5914de8cd915266e0325`.

Game assets have their own manifest, separate from model weights. `game-assets.json`
is an exact allowlist and SHA256 manifest, partitioned into shared, Castle,
Ancient Egypt and Racing resources. The runtime pack excludes original scene exports,
source terrain rasters, catalog preview images, recordings and conversion logs.
Raw authoring exports remain outside this source release.

The game runtime payload is 192,443,138 bytes (about 184 MiB). Source code and
runtime assets can be packaged independently with `scripts/package_release.py`.

Download only the game assets needed by a scene:

```bash
matrix-game download --manifest releases/game-assets.json \
  --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
  --revision 4419accdb7bb7881234c5914de8cd915266e0325 \
  --scene ancient-egypt --resources resources
```

## Model Files

`models.json` is the matching model manifest: three scene checkpoints, scene sink
images/contracts and shared encoder/codec files. The complete payload is about
44.2 GB in decimal units; each scene checkpoint is about 10 GB. Shared files are
downloaded once, not once per scene:

```bash
matrix-game download --manifest releases/models.json \
  --repo-id Skywork/Matrix-Game-4.0 --repo-type model \
  --revision 4419accdb7bb7881234c5914de8cd915266e0325 \
  --scene ancient-egypt --resources resources
matrix-game verify --manifest releases/models.json \
  --scene ancient-egypt --resources resources
```

Use `--scene castle`, `--scene racing` or `--scene all` as needed. Each scene
directory contains `model.safetensors` (inference tensors only), `sink.png` and
model metadata/configuration. Optimizer state is not distributed.
`scripts/export_checkpoint.py` exports the selected tensor dictionary and verifies
every tensor against the original before reporting success.

The Hub keeps the existing `fps-castle/`, `fps-egypt/` and `driving_game/` folders.
The manifest's `repo_path` fields map them to `resources/models/castle/`,
`resources/models/ancient-egypt/` and `resources/models/racing/` on disk, as
required by the scene profiles. The same filename need not have the same path
on the Hub and in the installed resource directory.

## Shared Dependencies

The T5 encoder, tokenizer and Wan2.2 VAE are downloaded from
[the official base model](https://huggingface.co/alibaba-pai/Wan2.2-Fun-5B-Control/tree/b8bc1a65ab71d054ba4636dc0dac104aa4df2686),
pinned to `b8bc1a65ab71d054ba4636dc0dac104aa4df2686`. The downloader uses each
entry's `repo_id`, `repo_type`, `revision` and `repo_path`, then verifies its
length and SHA256 before installation. No manual base-model download is needed.
The full TAEHV codec is hosted in the Matrix-Game repository.

After installation, shared files are under `resources/base/`:

```text
models_t5_umt5-xxl-enc-bf16.pth
google/umt5-xxl/{tokenizer.json,tokenizer_config.json,special_tokens_map.json,spiece.model}
Wan2.2_VAE.pth
taew2_2.pth
```

Use the exact shared resources listed in `models.json`; do not blindly replace
them with whatever upstream `main` contains at download time.
Do not substitute Wan2.1's 16-channel VAE. The release architecture is the
148-input-channel Fun 5B variant, not an unmodified TI2V backbone. The loader has
its fixed architecture fallback; an optional base `config.json` must describe
that exact model. Reference-image `ref_conv` weights are not used.

Model code, model weights and game assets each keep accompanying license and
attribution files. Shared upstream files should be pinned and deduplicated,
not copied separately into every scene bundle.

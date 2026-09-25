# WFF3D research code

WFF3D watermarks the Gaussian-generation path of feed-forward 3DGS models.
This repository contains the method and integration code for AnySplat,
DepthSplat, and YoNoSplat. It is a **source overlay**, not a standalone copy of
those projects. Pretrained weights, watermark extractor weights, datasets,
generated Gaussians, and experiment outputs are not included.

## Contents

| Path | Description |
| --- | --- |
| `wff3d/src/watermark/` | GADM, AAP, and 2D/3D attack utilities |
| `wff3d/fine_tune_anysplat.py` | AnySplat training and evaluation |
| `wff3d/save_init_encoder_decoder.py` | Export fixed AnySplat encoder and initial Gaussian head to CPU checkpoints |
| `wff3d/src/`, `wff3d/config/` | AnySplat integration and configuration overrides |
| `depthsplat/fine_tune_depthsplat.py` | DepthSplat training and evaluation |
| `YoNoSplat/fine_tune_yonosplat.py` | YoNoSplat training and evaluation |
| `depthsplat/src/dataset/`, `YoNoSplat/src/dataset/` | Raw DL3DV dataset adapters |
| `infer_anysplat_official_dl3dv.py` | AnySplat inference on DL3DV view-index JSONs |

The three `fine_tune_*.py` programs contain the model-level training loops;
`wff3d/src/watermark/gadm_aap.py` contains the shared method implementation.

## Environment

The development stack used Python 3.10, PyTorch 2.2.0, and CUDA 12.1. Create
an environment, install the matching CUDA build of PyTorch, then install the
additional Python dependencies:

```bash
conda env create -f environment.yml
conda activate wff3d
python -m pip install torch==2.2.0 torchvision==0.17.0 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Install the dependencies and CUDA extensions required by each upstream model
according to its repository. `requirements.txt` does not build those extensions.
For compiling them, use a CUDA 12.1 toolkit and point `CUDA_HOME` at that
toolkit, not at a different system `nvcc`. A GPU architecture unsupported by
PyTorch 2.2's extension builder requires a compatible newer stack; do not
silently substitute a different rasterizer implementation.

## Apply the overlay

Obtain the upstream [AnySplat](https://github.com/OpenRobotLab/AnySplat),
[DepthSplat](https://github.com/autonomousvision/depthsplat), and
[YoNoSplat](https://github.com/cvg/YoNoSplat) source trees. Keep their upstream
revisions fixed for reproducibility. Put the three working trees under one
parent directory and name the AnySplat-based working tree `wff3d`:

```bash
export WFF3D_CODE_DIR="$(pwd -P)"
export ANYSPLAT_DIR=/path/to/models/wff3d
export DEPTHSPLAT_DIR=/path/to/models/depthsplat
export YONOSPLAT_DIR=/path/to/models/YoNoSplat
cp -a "$WFF3D_CODE_DIR/wff3d/." "$ANYSPLAT_DIR/"
cp -a "$WFF3D_CODE_DIR/depthsplat/." "$DEPTHSPLAT_DIR/"
cp -a "$WFF3D_CODE_DIR/YoNoSplat/." "$YONOSPLAT_DIR/"
```

Run these commands from this repository's root. They replace files with
matching names but retain unlisted upstream files. Apply the overlay to clean
upstream checkouts, or review any local changes first. The DepthSplat and
YoNoSplat trainers import the shared watermark module from the sibling
`wff3d/src/watermark/` directory.

## Data and weights

Provide the three pretrained backbones, a compatible 48-bit render-space
watermark decoder, and DL3DV data separately. For AnySplat, configure the
training dataset root in `wff3d/config/dataset/dl3dv.yaml` (the default is
`./datasets/DL3DV-1K`). DepthSplat and YoNoSplat receive the raw DL3DV scene
root through `--dataset-root`. The evaluation view-index JSONs are also
external inputs; use the same index and target cameras when comparing models.

Prepare AnySplat's fixed encoder and Gaussian-head initialization after
installing its upstream dependencies:

```bash
cd "$ANYSPLAT_DIR"
python save_init_encoder_decoder.py --distill --out-dir checkpoints
```

The pretrained AnySplat model is fetched by `AnySplat.from_pretrained` if it
is not cached. The generated files are `encoder_fixed_cpu.pt` and
`decoder_init_cpu.pt`. The example watermark bit string in the CLI defaults is
public; choose a private `--wm-key` for any real ownership claim.

## Train

These are example 48-bit DL3DV settings. Set the paths to your own assets and
run each command from its corresponding overlaid working tree. Use `--help`
to inspect the full CLI, including checkpoint resume and attack options.

```bash
export WM_DECODER=/path/to/dec_48b_whit.torchscript.pt
export DL3DV_ROOT=/path/to/DL3DV
export OUTPUT_DIR=/path/to/outputs

cd "$ANYSPLAT_DIR"
python fine_tune_anysplat.py --config-path config/main.yaml \
  --experiment config/experiment/dl3dv.yaml \
  --wm_ckpt "$WM_DECODER" --distill --freeze-backbone \
  --gadm --aap --lambda-w 0.05 --lambda-gadm 0.05 --lambda-aap 0.05 \
  --max-steps 2000 --save-every 500 --output-dir "$OUTPUT_DIR/anysplat"

cd "$DEPTHSPLAT_DIR"
python fine_tune_depthsplat.py --ckpt /path/to/depthsplat.pth \
  --dataset-root "$DL3DV_ROOT" --wm-ckpt "$WM_DECODER" \
  --gadm --aap --lambda-w 0.02 --lambda-gadm 0.05 --lambda-aap 0.02 \
  --lpips --max-steps 2000 --save-every 500 \
  --output-dir "$OUTPUT_DIR/depthsplat"

cd "$YONOSPLAT_DIR"
python fine_tune_yonosplat.py --ckpt /path/to/yonosplat.ckpt \
  --dataset-root "$DL3DV_ROOT" --wm-ckpt "$WM_DECODER" \
  --gadm --aap --lambda-w 0.02 --lambda-gadm 0.05 --lambda-aap 0.02 \
  --lpips --max-steps 2000 --save-every 500 \
  --output-dir "$OUTPUT_DIR/yonosplat"
```

## Evaluate

For DepthSplat and YoNoSplat, run the same trainer with the base `--ckpt`,
`--dataset-root`, and `--wm-ckpt`, plus:

```bash
--eval-only --tuned-ckpt /path/to/step_2000.pt \
  --eval-index /path/to/6v_tgt8.json --eval-context-views 6 \
  --eval-batches 135
```

For AnySplat, either use its trainer's `--eval-only --tuned-ckpt` flags or run
the official-style DL3DV evaluation from this repository's root:

```bash
cd "$WFF3D_CODE_DIR"
export ANYSPLAT_ROOT="$ANYSPLAT_DIR"
python infer_anysplat_official_dl3dv.py \
  --data-root "$DL3DV_ROOT" --index-root /path/to/eval-index-jsons \
  --tags 6v_tgt8 --pose-mode vggt --image-shape 448x448 \
  --tuned-ckpt "$OUTPUT_DIR/anysplat/checkpoints/step_2000.pt" \
  --wm-ckpt "$WM_DECODER" --output-root "$OUTPUT_DIR/anysplat/eval"
```

The `--image-shape` above matches the upstream AnySplat inference path;
the AnySplat training config specifies its training shape separately. Keep
the same view-index file, image source, and metric settings across comparisons.

## Publication scope

This overlay does not redistribute model or decoder weights. The modified
upstream source files retain their original project licenses; see
`THIRD_PARTY_LICENSES.md`. Preserve those notices when publishing or
redistributing the code. Checkpoint access and dataset licenses are governed
by their respective providers. A license for original WFF3D contributions
has not been selected in this package; choose one before a public release.

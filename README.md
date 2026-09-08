# videoRetrieval

Run Apple's **DepthPro** monocular depth-estimation model
([`apple/DepthPro-hf`](https://huggingface.co/apple/DepthPro-hf)) on local
image(s) using Hugging Face `transformers`.

DepthPro predicts a **metric** depth map (values in metres) plus an estimated
focal length / field of view from a single RGB image.

## Setup

Uses the system Python at `/usr/local/bin/python` (RunPod `runpod-torch-v280`
image), which already ships `torch`/`torchvision 2.8.0+cu128` with working CUDA.
Only the Hugging Face stack is added on top:

```bash
/usr/local/bin/python -m pip install -r requirements.txt
```

The model weights (~1.8 GB) are cached on the **local container disk** at
`~/.cache/huggingface`. `run_depth.py` forces this automatically — the RunPod
image points `HF_HOME` at the network volume, where loading the model takes
~100 s instead of ~10 s. Override with `DEPTHPRO_HF_HOME=/some/path` if needed.
`HF_HUB_OFFLINE=1` is set by the script since the weights are already local.

## Usage

```bash
# grab a test image (optional) -> images/sample.jpg
/usr/local/bin/python fetch_sample.py

# one image
/usr/local/bin/python run_depth.py --image images/sample.jpg

# many images in one run (weights load once)
/usr/local/bin/python run_depth.py --image a.jpg --image b.png
/usr/local/bin/python run_depth.py --input-dir images --outdir outputs
```

### Serve mode — fastest for repeated / streaming use

Loads the model once, then reads image paths from stdin (one per line) and
processes each at ~0.5 s. Avoids paying the ~90 s startup (Python + torch import
+ weight load + warm-up) on every image.

```bash
/usr/local/bin/python run_depth.py --serve
# then type / pipe paths:
images/sample.jpg
/data/frame_0001.png
<empty line or Ctrl-D to quit>
```

Or drive it from another process:

```bash
find /data/frames -name '*.png' | /usr/local/bin/python run_depth.py --serve --outdir out
```

### Outputs (in `--outdir`, default `./outputs/`)

| file | contents |
|------|----------|
| `<name>_depth_raw.npy`   | metric depth in metres, `float32`, shape `(H, W)` |
| `<name>_depth_gray.png`  | grayscale preview (near = bright, far = dark) |
| `<name>_depth_color.png` | `inferno` colormap preview |

### Options

```
--image PATH        image file; repeat for several
--input-dir DIR     process every image in DIR
--serve             stdin path loop, model stays resident
--outdir DIR        output directory (default: outputs)
--model ID          HF model id (default: apple/DepthPro-hf)
--device {auto,cuda,cpu}
--no-warmup         skip the warm-up pass
--no-color          skip the colormap PNG
--tune              enable cuDNN autotuning: ~2 min up front, ~0.1 s/image
                    faster afterwards; only worth it for hundreds of images
```

## Speed

Measured on the RTX A4000 (`transformers 5.16.1`):

| stage | time | notes |
|-------|------|-------|
| Python + torch/transformers import | ~60 s | per process, unavoidable |
| Model load (local cache) | ~10–20 s | ~100 s if `HF_HOME` is on the network volume |
| Warm-up pass | ~7 s | one-time CUDA kernel load |
| **Inference** | **~0.5 s / image** | was ~7 s before TF32 + SDPA + `inference_mode` + fast processor + warm model |

So: use `--input-dir` or `--serve` to amortise the ~90 s startup across many
images instead of paying it each time.

Enabled optimisations: `float16`, TF32 matmul, SDPA attention, `channels_last`,
`torch.inference_mode()`, the torchvision (GPU) image processor, and a resident
warm model. cuDNN autotuning is opt-in (`--tune`) because DepthPro's many unique
conv shapes make it a ~2 min one-time cost.

## Notes

- DepthPro's absolute scale can be off on unusual framing/crops; the relative
  depth structure is what's reliable.
- On CPU it falls back to `float32` and is very slow.

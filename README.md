# videoRetrieval

Run Apple's **DepthPro** monocular depth-estimation model
([`apple/DepthPro-hf`](https://huggingface.co/apple/DepthPro-hf)) on a local
image using Hugging Face `transformers`.

DepthPro predicts a **metric** depth map (values in metres) plus an estimated
focal length / field of view, from a single RGB image, in well under a second on
this box's RTX A4000 (first call is slower — CUDA warm-up + weight load).

## Setup

Uses the system Python at `/usr/local/bin/python` (RunPod `runpod-torch-v280`
image), which already ships `torch 2.8.0+cu128` with working CUDA. Only the
Hugging Face stack is added on top:

```bash
/usr/local/bin/python -m pip install -r requirements.txt
```

`requirements.txt` pins `torch` to the version already installed, so pip won't
reinstall it.

### Keep the model cache on the persistent volume

The weights are ~1.9 GB. The container disk is small and wiped on restart, so
point the HF cache at `/workspace`:

```bash
export HF_HOME=/workspace/hf_cache
```

## Usage

```bash
export HF_HOME=/workspace/hf_cache

# grab a test image (optional) -> images/sample.jpg
/usr/local/bin/python fetch_sample.py

# run depth estimation
/usr/local/bin/python run_depth.py --image images/sample.jpg
/usr/local/bin/python run_depth.py --image /path/to/your/photo.jpg --outdir outputs
```

### Outputs (in `--outdir`, default `./outputs/`)

| file | contents |
|------|----------|
| `<name>_depth_raw.npy`   | metric depth in metres, `float32`, shape `(H, W)` |
| `<name>_depth_gray.png`  | grayscale preview (near = bright, far = dark) |
| `<name>_depth_color.png` | `inferno` colormap preview |

The script also prints the depth range and the model's estimated focal length
and horizontal field of view.

### Options

```
--image   PATH   local image file (required)
--outdir  DIR    output directory (default: outputs)
--model   ID     HF model id (default: apple/DepthPro-hf)
--device {auto,cuda,cpu}
```

## Notes

- On CUDA the model runs in `float16`; on CPU it falls back to `float32` (slow).
- Verified with `transformers 5.16.1`; `run_depth.py` also handles the older
  `torch_dtype` kwarg for `transformers < 4.56`.
- DepthPro's absolute scale can be off on unusual framing/crops; the relative
  depth structure is what's reliable.

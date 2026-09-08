"""Run Apple's DepthPro monocular depth-estimation model on local image(s).

Model card: https://huggingface.co/apple/DepthPro-hf

Speed notes:
  * The model weights load once, then every image reuses the warm model, so
    pass many images in a single invocation rather than re-running the script.
  * Keep the HF cache on the local container disk (fast), not the network
    volume. This script defaults HF_HOME to ~/.cache/huggingface unless you
    set it yourself.
  * TF32 / cuDNN autotuning / SDPA attention / the fast (GPU) image processor
    are all enabled below.

Examples:
    python run_depth.py --image images/sample.jpg
    python run_depth.py --image a.jpg --image b.jpg --image c.png
    python run_depth.py --input-dir images --outdir outputs

Outputs per image (in --outdir, default ./outputs):
    <name>_depth_raw.npy    metric depth in metres, float32, shape (H, W)
    <name>_depth_gray.png   near = bright, far = dark
    <name>_depth_color.png  inferno colormap for quick viewing
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Force the HF cache onto the local container disk BEFORE transformers is
# imported. The RunPod image pre-sets HF_HOME to the network volume, which
# makes model load take ~100s instead of ~3s. Override with DEPTHPRO_HF_HOME.
os.environ["HF_HOME"] = os.environ.get(
    "DEPTHPRO_HF_HOME", str(Path.home() / ".cache" / "huggingface")
)
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # weights are already cached locally

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "apple/DepthPro-hf"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", action="append", default=[], type=Path,
                   help="Image file. Repeat for multiple images.")
    p.add_argument("--input-dir", type=Path, default=None,
                   help="Process every image in this directory.")
    p.add_argument("--serve", action="store_true",
                   help="Load the model once, then read image paths from stdin "
                        "(one per line) and process each. Avoids paying startup "
                        "(~90s) per image. Ctrl-D / empty line to stop.")
    p.add_argument("--outdir", type=Path, default=Path("outputs"), help="Output directory.")
    p.add_argument("--model", default=MODEL_ID, help=f"HF model id (default: {MODEL_ID}).")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-warmup", action="store_true",
                   help="Skip the warm-up pass (first real image will be slower).")
    p.add_argument("--tune", action="store_true",
                   help="Enable cuDNN autotuning. Adds a big one-time cost (~2 min) "
                        "but shaves ~0.1s/image; only worth it for hundreds of images.")
    p.add_argument("--no-color", action="store_true", help="Skip the colormap PNG.")
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def collect_images(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = list(args.image)
    if args.input_dir is not None:
        paths += sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    seen, out = set(), []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        if not p.is_file():
            raise SystemExit(f"Image not found: {p}")
        out.append(p)
    if not out:
        raise SystemExit("No images given. Use --image PATH (repeatable) or --input-dir DIR.")
    return out


def save_outputs(depth: np.ndarray, stem: str, outdir: Path, want_color: bool) -> None:
    np.save(outdir / f"{stem}_depth_raw.npy", depth)
    inv = 1.0 / np.clip(depth, 1e-6, None)
    inv_norm = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)
    Image.fromarray((inv_norm * 255).astype(np.uint8), mode="L").save(outdir / f"{stem}_depth_gray.png")
    if want_color:
        try:
            import matplotlib.cm as cm
            rgb = (cm.inferno(inv_norm)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(rgb).save(outdir / f"{stem}_depth_color.png")
        except Exception as exc:  # noqa: BLE001
            print(f"  (skipped colormap image: {exc})")


def infer_one(model, processor, device, dtype, path: Path, outdir: Path, want_color: bool) -> float:
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, dtype)

    ti = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - ti

    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]
    depth = post["predicted_depth"].to(torch.float32).cpu().numpy()
    focal = post.get("focal_length")
    fov = post.get("field_of_view")

    print(f"  {path.name}  {image.width}x{image.height}  infer {dt:.2f}s  "
          f"depth {depth.min():.2f}-{depth.max():.2f}m (median {np.median(depth):.2f}m)"
          + (f"  focal {float(focal):.0f}px" if focal is not None else "")
          + (f"  fov {float(fov):.1f}deg" if fov is not None else ""))

    save_outputs(depth, path.stem, outdir, want_color=want_color)
    return dt


def main() -> None:
    args = parse_args()
    images = [] if args.serve else collect_images(args)

    device = pick_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # DepthPro hits many unique conv shapes; autotuning them costs ~2 min
        # up front, so it is opt-in via --tune.
        torch.backends.cudnn.benchmark = args.tune

    print(f"Device: {device} | dtype: {dtype} | model: {args.model} | HF_HOME={os.environ['HF_HOME']}")
    if not args.serve:
        print(f"{len(images)} image(s) to process")

    t0 = time.perf_counter()
    try:
        processor = AutoImageProcessor.from_pretrained(args.model, backend="torchvision")
    except (TypeError, ValueError):
        processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    try:
        model = AutoModelForDepthEstimation.from_pretrained(
            args.model, dtype=dtype, attn_implementation="sdpa"
        )
    except (TypeError, ValueError):
        model = AutoModelForDepthEstimation.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()
    if device.type == "cuda":
        model.to(memory_format=torch.channels_last)
    print(f"Model load: {time.perf_counter() - t0:.2f}s")

    args.outdir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda" and not args.no_warmup:
        tw = time.perf_counter()
        dummy = Image.new("RGB", (768, 768))
        winputs = processor(images=dummy, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            model(**winputs)
        torch.cuda.synchronize()
        print(f"Warm-up: {time.perf_counter() - tw:.2f}s")

    want_color = not args.no_color

    if args.serve:
        import sys
        print("serve: ready. Enter image paths on stdin (empty line / Ctrl-D to quit).", flush=True)
        for line in sys.stdin:
            p = Path(line.strip())
            if not line.strip():
                break
            if not p.is_file():
                print(f"  ! not found: {p}", flush=True)
                continue
            try:
                infer_one(model, processor, device, dtype, p, args.outdir, want_color)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! failed on {p}: {exc}", flush=True)
            print("", flush=True)
        print("serve: done.")
        return

    total_infer = 0.0
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}]", end="")
        total_infer += infer_one(model, processor, device, dtype, path, args.outdir, want_color)

    n = len(images)
    print(f"\nDone. Inference: {total_infer:.2f}s total, {total_infer / n:.2f}s/image "
          f"({n} image(s)). Outputs in {args.outdir}/")


if __name__ == "__main__":
    main()

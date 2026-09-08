"""Run Apple's DepthPro monocular depth-estimation model on a local image.

Model card: https://huggingface.co/apple/DepthPro-hf

Example:
    python run_depth.py --image images/sample.jpg
    python run_depth.py --image /path/to/photo.png --outdir outputs

Outputs (written to --outdir, default ./outputs):
    <name>_depth_raw.npy    metric depth in metres, float32, shape (H, W)
    <name>_depth_gray.png   near = white, far = black
    <name>_depth_color.png  inferno colormap for quick viewing
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "apple/DepthPro-hf"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path, help="Path to a local image file.")
    p.add_argument("--outdir", type=Path, default=Path("outputs"), help="Directory for output files.")
    p.add_argument("--model", default=MODEL_ID, help=f"HF model id (default: {MODEL_ID}).")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device (default: auto).",
    )
    return p.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def main() -> None:
    args = parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    device = pick_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype} | model: {args.model}")

    image = Image.open(args.image).convert("RGB")
    print(f"Input image: {args.image}  ({image.width}x{image.height})")

    print("Loading processor + model (first run downloads ~1.9 GB of weights)...")
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    inputs = processor(images=image, return_tensors="pt").to(device, dtype)

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"Inference: {time.perf_counter() - t0:.2f}s")

    post = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]

    depth = post["predicted_depth"].to(torch.float32).cpu().numpy()  # metres, (H, W)
    focal_px = float(post["focal_length"]) if post.get("focal_length") is not None else None
    fov_deg = float(post["field_of_view"]) if post.get("field_of_view") is not None else None

    print(f"Depth range: {depth.min():.2f} m .. {depth.max():.2f} m  (median {np.median(depth):.2f} m)")
    if focal_px is not None:
        print(f"Estimated focal length: {focal_px:.1f} px")
    if fov_deg is not None:
        print(f"Estimated horizontal field of view: {fov_deg:.1f} deg")

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    raw_path = args.outdir / f"{stem}_depth_raw.npy"
    np.save(raw_path, depth)

    # Normalised inverse depth reads better visually (foreground detail preserved).
    inv = 1.0 / np.clip(depth, 1e-6, None)
    inv_norm = (inv - inv.min()) / (inv.max() - inv.min() + 1e-8)

    gray_path = args.outdir / f"{stem}_depth_gray.png"
    Image.fromarray((inv_norm * 255).astype(np.uint8), mode="L").save(gray_path)

    color_path = args.outdir / f"{stem}_depth_color.png"
    try:
        import matplotlib.cm as cm

        rgb = (cm.inferno(inv_norm)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(color_path)
    except Exception as exc:  # noqa: BLE001 - visualisation is optional
        color_path = None
        print(f"(skipped colormap image: {exc})")

    print("\nWrote:")
    print(f"  {raw_path}")
    print(f"  {gray_path}")
    if color_path is not None:
        print(f"  {color_path}")


if __name__ == "__main__":
    main()

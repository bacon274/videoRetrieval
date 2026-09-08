"""Extract 2D and 3D hand keypoints from an image with HaMeR.

This script follows the detector -> ViTPose -> HaMeR pipeline used by the
HaMeR demo, but writes keypoints instead of rendering a video. The HaMeR
repository must be installed or available on PYTHONPATH.

Example:
    python extract_hamer_keypoints.py --image images/hand.jpg --outdir outputs

The output is a compressed NumPy archive containing one entry per detected
hand. See ``--help`` for the array names and detector options.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", required=True, type=Path, help="Input image path.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--checkpoint", default=None, help="HaMeR checkpoint path or URL."
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--body-detector", choices=("vitdet", "regnety"), default="vitdet"
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=2.0,
        help="Scale around each ViTPose hand box before HaMeR inference.",
    )
    return parser.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return torch.device(choice)


def load_detector(body_detector: str):
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    if body_detector == "vitdet":
        import hamer
        from detectron2.config import LazyConfig

        cfg_path = (
            Path(hamer.__file__).parent
            / "configs"
            / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )
        config = LazyConfig.load(str(cfg_path))
        config.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for predictor in config.model.roi_heads.box_predictors:
            predictor.test_score_thresh = 0.25
    else:
        from detectron2 import model_zoo

        config = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        config.model.roi_heads.box_predictor.test_score_thresh = 0.5
        config.model.roi_heads.box_predictor.test_nms_thresh = 0.4

    return DefaultPredictor_Lazy(config)


def detect_hand_boxes(
    image_rgb: np.ndarray, detector, pose_model
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    detections = detector(image_rgb)["instances"]
    valid = (detections.pred_classes == 0) & (detections.scores > 0.5)
    person_boxes = detections.pred_boxes.tensor[valid].cpu().numpy()
    person_scores = detections.scores[valid].cpu().numpy()
    if len(person_boxes) == 0:
        raise RuntimeError("No person was detected in the image.")

    pose_outputs = pose_model.predict_pose(
        image_rgb[:, :, ::-1],
        [np.concatenate([person_boxes, person_scores[:, None]], axis=1)],
    )

    hand_boxes = []
    is_right = []
    for pose in pose_outputs:
        for keypoints, right in (
            (pose["keypoints"][-42:-21], 0),
            (pose["keypoints"][-21:], 1),
        ):
            confident = keypoints[:, 2] > 0.5
            if confident.sum() > 3:
                hand_boxes.append(
                    [
                        keypoints[confident, 0].min(),
                        keypoints[confident, 1].min(),
                        keypoints[confident, 0].max(),
                        keypoints[confident, 1].max(),
                    ]
                )
                is_right.append(right)

    if not hand_boxes:
        raise RuntimeError("No hands were detected in the image.")
    return (
        np.asarray(hand_boxes, dtype=np.float32),
        np.asarray(is_right, dtype=np.int64),
        pose_outputs,
    )


def project_keypoints(
    keypoints_3d: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    width: int,
    height: int,
) -> np.ndarray:
    points = keypoints_3d + camera_translation[:, None, :]
    safe_z = np.maximum(points[..., 2], 1e-6)
    projected = np.empty((*points.shape[:-1], 2), dtype=np.float32)
    projected[..., 0] = points[..., 0] * focal_length / safe_z + width / 2
    projected[..., 1] = points[..., 1] * focal_length / safe_z + height / 2
    return projected


def run(args: argparse.Namespace) -> Path:
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
    from hamer.datasets.vitdet_dataset import ViTDetDataset
    from hamer.utils import recursive_to
    from hamer.utils.renderer import cam_crop_to_full
    from vitpose_model import ViTPoseModel

    device = pick_device(args.device)

    image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINT
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(checkpoint)
    model = model.to(device).eval()

    detector = load_detector(args.body_detector)
    pose_model = ViTPoseModel(device)

    boxes, right, vitpose_outputs = detect_hand_boxes(image_rgb, detector, pose_model)

    dataset = ViTDetDataset(
        model_cfg, image_rgb, boxes, right, rescale_factor=args.rescale_factor
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

    vertices, joints_3d, camera_translations = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = recursive_to(batch, device)
            output = model(batch)

            pred_cam = output["pred_cam"]
            multiplier = 2 * batch["right"] - 1
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            image_size = batch["img_size"].float()

            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH
                / model_cfg.MODEL.IMAGE_SIZE
                * image_size.max()
            )
            camera_translation = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                image_size,
                scaled_focal_length,
            )

            batch_vertices = output["pred_vertices"].detach().cpu().numpy()
            batch_joints = output["pred_keypoints_3d"].detach().cpu().numpy()
            batch_right = batch["right"].detach().cpu().numpy()
            batch_vertices[:, :, 0] *= 2 * batch_right[:, None] - 1
            batch_joints[:, :, 0] *= 2 * batch_right[:, None] - 1
            vertices.extend(batch_vertices)
            joints_3d.extend(batch_joints)
            camera_translations.extend(camera_translation.detach().cpu().numpy())

    vertices = np.asarray(vertices, dtype=np.float32)
    joints_3d = np.asarray(joints_3d, dtype=np.float32)
    camera_translations = np.asarray(camera_translations, dtype=np.float32)

    focal_length = (
        float(model_cfg.EXTRA.FOCAL_LENGTH)
        / model_cfg.MODEL.IMAGE_SIZE
        * max(width, height)
    )
    joints_2d = project_keypoints(
        joints_3d, camera_translations, focal_length, width, height
    )
    vitpose_keypoints_2d = np.asarray(
        [pose["keypoints"] for pose in vitpose_outputs], dtype=np.float32
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    output_path = args.outdir / f"{args.image.stem}_hamer_keypoints.npz"
    np.savez_compressed(
        output_path,
        hamer_keypoints_3d=joints_3d,
        hamer_keypoints_2d=joints_2d,
        vertices=vertices,
        camera_translation=camera_translations,
        hand_boxes=boxes,
        is_right=right,
        vitpose_keypoints_2d=vitpose_keypoints_2d,
        image_size=np.asarray([width, height], dtype=np.int32),
    )

    metadata = {
        "image": str(args.image),
        "num_hands": int(len(boxes)),
        "keypoint_convention": "HaMeR pred_keypoints_3d / MANO joints plus fingertips",
        "focal_length_pixels": focal_length,
    }
    (args.outdir / f"{args.image.stem}_hamer_keypoints.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Detected {len(boxes)} hand(s). Saved {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except ImportError as exc:
        raise SystemExit(
            "HaMeR dependencies are unavailable. Install HaMeR, Detectron2, "
            "ViTPose, and their model assets before running this script.\n"
            f"Import error: {exc}"
        ) from exc


if __name__ == "__main__":
    main()

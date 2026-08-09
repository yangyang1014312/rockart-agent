import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from mmdet.apis import inference_detector, init_detector

try:
    from pycocotools import mask as mask_utils
except ImportError:  # pragma: no cover - only used when masks are requested
    mask_utils = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG = os.getenv("MMDET_CONFIG", str(BASE_DIR / "configs" / "rtmdet_ins_l_rock_art.py"))
CHECKPOINT = os.getenv(
    "MMDET_CHECKPOINT",
    str(BASE_DIR / "checkpoints" / "rockart.pth"),
)
DEVICE = os.getenv("MMDET_DEVICE", "cuda:0")
DEFAULT_SCORE_THR = float(os.getenv("SCORE_THR", "0.3"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

model = None


def _class_names() -> list[str]:
    dataset_meta = getattr(model, "dataset_meta", {}) if model is not None else {}
    classes = dataset_meta.get("classes", [])
    return list(classes)


def _encode_masks(masks: Any) -> list[dict[str, Any]]:
    if mask_utils is None:
        raise HTTPException(
            status_code=500,
            detail="pycocotools is required for include_masks=true.",
        )

    encoded_masks = []
    masks_np = masks.cpu().numpy() if hasattr(masks, "cpu") else np.asarray(masks)
    for binary_mask in masks_np.astype(np.uint8):
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("ascii")
        encoded_masks.append(rle)
    return encoded_masks


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not Path(CONFIG).is_file():
        raise RuntimeError(f"Config file not found: {CONFIG}")
    if not Path(CHECKPOINT).is_file():
        raise RuntimeError(f"Checkpoint file not found: {CHECKPOINT}")

    model = init_detector(CONFIG, CHECKPOINT, device=DEVICE)
    yield


app = FastAPI(title="Rock Art Detection API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE,
        "config": CONFIG,
        "checkpoint": CHECKPOINT,
        "classes": _class_names(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    score_thr: float = Query(DEFAULT_SCORE_THR, ge=0.0, le=1.0),
    include_masks: bool = Query(False),
) -> dict[str, Any]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {suffix or '<none>'}",
        )

    image_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            image_path = tmp.name

        result = inference_detector(model, image_path)
        pred = result.pred_instances

        keep = pred.scores >= score_thr
        pred = pred[keep]

        bboxes = pred.bboxes.cpu().numpy().tolist()
        scores = pred.scores.cpu().numpy().tolist()
        labels = pred.labels.cpu().numpy().tolist()
        classes = _class_names()

        detections = []
        for bbox, score, label in zip(bboxes, scores, labels):
            item = {
                "bbox": bbox,
                "score": score,
                "label": label,
                "class_name": classes[label] if label < len(classes) else str(label),
            }
            detections.append(item)

        response = {
            "count": len(detections),
            "score_thr": score_thr,
            "detections": detections,
            "bboxes": bboxes,
            "scores": scores,
            "labels": labels,
        }

        if include_masks and hasattr(pred, "masks"):
            response["masks"] = _encode_masks(pred.masks)

        return response
    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

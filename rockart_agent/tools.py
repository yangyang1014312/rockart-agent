"""Tools that wrap the RTMDet-Ins FastAPI service."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = os.getenv("ROCKART_API_URL", "http://127.0.0.1:8000")


@dataclass(slots=True)
class Detection:
    bbox: list[float]
    score: float
    label: int
    class_name: str
    mask: dict[str, Any] | None = None


@dataclass(slots=True)
class SegmentationReport:
    image_path: str
    api_url: str
    score_thr: float
    count: int
    detections: list[Detection] = field(default_factory=list)
    masks: list[dict[str, Any]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def class_counts(self) -> dict[str, int]:
        return dict(Counter(det.class_name for det in self.detections))

    @property
    def top_detection(self) -> Detection | None:
        return max(self.detections, key=lambda item: item.score, default=None)

    @property
    def max_score(self) -> float:
        top = self.top_detection
        return top.score if top is not None else 0.0

    def summary(self) -> str:
        if not self.detections:
            return f"No instances found in {self.image_path} at score_thr={self.score_thr:.2f}."

        class_bits = ", ".join(f"{name}: {count}" for name, count in self.class_counts.items())
        top = self.top_detection
        top_text = ""
        if top is not None:
            top_text = f" Top detection: {top.class_name} ({top.score:.3f})."
        return f"Detected {self.count} instance(s) in {self.image_path}. {class_bits}.{top_text}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "api_url": self.api_url,
            "score_thr": self.score_thr,
            "count": self.count,
            "class_counts": self.class_counts,
            "max_score": self.max_score,
            "detections": [
                {
                    "bbox": det.bbox,
                    "score": det.score,
                    "label": det.label,
                    "class_name": det.class_name,
                    **({"mask": det.mask} if det.mask is not None else {}),
                }
                for det in self.detections
            ],
            "masks": self.masks,
            "raw": self.raw,
            "summary": self.summary(),
        }


class RockArtDetectionTool:
    """Callable tool for the rock art instance segmentation API."""

    def __init__(self, api_url: str = DEFAULT_API_URL, timeout: float = 120.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.api_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def analyze_image(
        self,
        image_path: str | Path,
        score_thr: float = 0.3,
        include_masks: bool = False,
    ) -> SegmentationReport:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        with path.open("rb") as handle:
            response = self.session.post(
                f"{self.api_url}/predict",
                files={"file": (path.name, handle, "application/octet-stream")},
                params={"score_thr": score_thr, "include_masks": include_masks},
                timeout=self.timeout,
            )

        response.raise_for_status()
        payload = response.json()

        detections: list[Detection] = []
        masks = payload.get("masks")
        for index, item in enumerate(payload.get("detections", [])):
            detections.append(
                Detection(
                    bbox=[float(value) for value in item.get("bbox", [])],
                    score=float(item.get("score", 0.0)),
                    label=int(item.get("label", -1)),
                    class_name=str(item.get("class_name", "unknown")),
                    mask=(masks[index] if isinstance(masks, list) and index < len(masks) else item.get("mask")),
                )
            )

        return SegmentationReport(
            image_path=str(path),
            api_url=self.api_url,
            score_thr=float(payload.get("score_thr", score_thr)),
            count=int(payload.get("count", len(detections))),
            detections=detections,
            masks=masks,
            raw=payload,
        )

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "RockArtDetectionTool":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def analyze_rock_art_image(
    image_path: str,
    api_url: str = DEFAULT_API_URL,
    score_thr: float = 0.3,
    include_masks: bool = False,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Agent tool function: run instance segmentation and return structured JSON."""

    with RockArtDetectionTool(api_url=api_url, timeout=timeout) as tool:
        report = tool.analyze_image(
            image_path=image_path,
            score_thr=score_thr,
            include_masks=include_masks,
        )
    return {"status": "success", **report.to_dict()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Call the Rock Art segmentation API.")
    parser.add_argument("image", help="Path to the image to analyze.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Base URL of the FastAPI service.")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Confidence threshold.")
    parser.add_argument("--include-masks", action="store_true", help="Request COCO RLE masks.")
    parser.add_argument("--json", action="store_true", help="Print the full structured result.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = analyze_rock_art_image(
        image_path=args.image,
        api_url=args.api_url,
        score_thr=args.score_thr,
        include_masks=args.include_masks,
    )
    if not args.json:
        print(result["summary"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

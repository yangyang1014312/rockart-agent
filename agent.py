"""Backward-compatible entrypoint for the rock art detection tool."""

from rockart_agent.tools import (
    DEFAULT_API_URL,
    Detection,
    RockArtDetectionTool,
    SegmentationReport,
    analyze_rock_art_image,
    main,
)

RockArtSegmentationAgent = RockArtDetectionTool

__all__ = [
    "DEFAULT_API_URL",
    "Detection",
    "RockArtDetectionTool",
    "RockArtSegmentationAgent",
    "SegmentationReport",
    "analyze_rock_art_image",
]


if __name__ == "__main__":
    main()

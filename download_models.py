"""Download the MediaPipe models into ./models (only needed if they are missing)."""
from __future__ import annotations

import urllib.request

from retrolens import config

MODELS = {
    config.HAND_MODEL:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    config.SEGMENT_MODEL:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}


def main() -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in MODELS.items():
        if path.exists():
            print(f"ok       {path.name}")
            continue
        print(f"download {path.name} ...")
        urllib.request.urlretrieve(url, path)
    print("done")


if __name__ == "__main__":
    main()

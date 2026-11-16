"""Retrolens 2.0 - run with:  python main.py"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from retrolens import config
from retrolens.app import App

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class FrameSource:
    """Webcam by default; a video or image file can be used instead (handy for testing)."""

    def __init__(self, camera: int, width: int, height: int, source: str = ""):
        self._still = None
        self._cap = None
        self._loop = bool(source)
        if source and source.lower().endswith(IMAGE_EXT):
            self._still = cv2.imread(source)
            if self._still is None:
                raise RuntimeError(f"Could not read image: {source}")
            return
        self._cap = cv2.VideoCapture(source if source else camera)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open {'file ' + source if source else 'camera ' + str(camera)}.\n"
                "On macOS, allow camera access for your terminal in "
                "System Settings > Privacy & Security > Camera, or try --camera 1."
            )
        if not source:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        if self._still is not None:
            return self._still.copy()
        ok, frame = self._cap.read()
        if not ok and self._loop:                       # rewind video files
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrolens 2.0 - hand-tracking portal filters")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--source", default="", help="video or image file to use instead of the camera")
    parser.add_argument("--no-mirror", action="store_true", help="do not flip the image horizontally")
    args = parser.parse_args()

    try:
        source = FrameSource(args.camera, args.width, args.height, args.source)
        app = App(mirror=not args.no_mirror)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[retrolens] {exc}", file=sys.stderr)
        return 1

    print("[retrolens] running - press H for help, Q or ESC to quit")
    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    start = time.monotonic()
    sized = False
    try:
        while True:
            frame = source.read()
            if frame is None:
                print("[retrolens] no frame from the source, exiting", file=sys.stderr)
                break
            out = app.process(frame, time.monotonic() - start)
            if not sized:
                cv2.resizeWindow(config.WINDOW_TITLE, out.shape[1], out.shape[0])
                sized = True
            cv2.imshow(config.WINDOW_TITLE, out)
            key = cv2.waitKey(1) & 0xFF
            if key != 255 and not app.handle_key(key):
                break
            if cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break                                    # window closed with the mouse
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        source.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

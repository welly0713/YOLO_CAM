"""Camera reader that continuously keeps the newest frame available."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatestFrame:
    """A frame and the monotonic time at which it was received."""

    image: object
    received_at: float


class LatestFrameReader:
    """Read an RTSP stream in the background and expose only its latest frame.

    Reading and inference happen in separate threads.  This prevents the
    detector from gradually falling behind a live camera stream while YOLO is
    processing a frame.
    """

    def __init__(
        self,
        source: str | int,
        reconnect_delay_seconds: float = 5.0,
        open_timeout_seconds: float = 10.0,
        transport: str = "tcp",
        frame_width: int | None = None,
        frame_height: int | None = None,
        frame_fps: float = 0.0,
    ) -> None:
        self.source = source
        self.reconnect_delay_seconds = max(0.1, reconnect_delay_seconds)
        self.open_timeout_seconds = max(0.1, open_timeout_seconds)
        self.transport = transport
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_fps = max(0.0, frame_fps)
        self.is_webcam = isinstance(source, int)

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: LatestFrame | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="camera-reader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def latest(self) -> LatestFrame | None:
        """Return a copy of the newest frame, or ``None`` before first read."""

        with self._lock:
            if self._latest is None:
                return None
            # The capture thread may overwrite the frame while inference is
            # using it, so copy the image before returning it.
            return LatestFrame(self._latest.image.copy(), self._latest.received_at)

    def _open_capture(self) -> cv2.VideoCapture | None:
        if self.is_webcam:
            # DirectShow generally gives the most reliable webcam behavior on
            # Windows.  Fall back to OpenCV's automatic backend if needed.
            backend = (
                getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
                if os.name == "nt"
                else cv2.CAP_ANY
            )
            capture = cv2.VideoCapture(self.source, backend)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.source)
        else:
            # FFmpeg reads this option when VideoCapture is created, so it must
            # be set before opening the RTSP URL.
            if self.transport:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                    f"rtsp_transport;{self.transport}"
                )
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not capture.isOpened():
                capture.release()
                # The fallback is useful on machines where OpenCV was built
                # without the FFmpeg backend.
                capture = cv2.VideoCapture(self.source)

        if not capture.isOpened():
            capture.release()
            return None

        if self.is_webcam:
            if self.frame_width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            if self.frame_height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            if self.frame_fps > 0:
                capture.set(cv2.CAP_PROP_FPS, self.frame_fps)

        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(self.open_timeout_seconds * 1000),
            )
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(self.open_timeout_seconds * 1000),
            )
        return capture

    def _run(self) -> None:
        while not self._stop_event.is_set():
            capture = self._open_capture()
            if capture is None:
                LOGGER.warning(
                    "無法開啟影像來源，%.1f 秒後重試。",
                    self.reconnect_delay_seconds,
                )
                self._stop_event.wait(self.reconnect_delay_seconds)
                continue

            if self.is_webcam:
                LOGGER.info("本機鏡頭 index=%s 已連線。", self.source)
            else:
                LOGGER.info("RTSP 串流已連線（transport=%s）。", self.transport)
            try:
                while not self._stop_event.is_set():
                    ok, image = capture.read()
                    if not ok or image is None:
                        LOGGER.warning("讀取影像失敗，準備重新連線。")
                        break
                    with self._lock:
                        self._latest = LatestFrame(image, time.monotonic())
            finally:
                capture.release()

            if not self._stop_event.is_set():
                self._stop_event.wait(self.reconnect_delay_seconds)

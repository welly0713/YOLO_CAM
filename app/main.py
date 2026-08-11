"""Program entry point for RTSP person monitoring."""

from __future__ import annotations

import logging
import signal
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from .camera import LatestFrameReader
from .config import Settings
from .detection import AlertGate, DetectionWindow
from .notifier import Notifier


LOGGER = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _count_people(
    model: YOLO,
    image: object,
    confidence: float,
    device: str,
    image_size: int,
    show_preview: bool = False,
) -> tuple[int, object]:
    """Run YOLO and return the person count plus an optional annotated frame.

    COCO class 0 is ``person``.  Restricting inference to that class avoids
    doing unnecessary work for objects that are irrelevant to this monitor.
    """

    results = model.predict(
        source=image,
        conf=confidence,
        classes=[0],
        imgsz=image_size,
        device=device,
        verbose=False,
    )
    boxes = results[0].boxes if results else None
    person_count = 0 if boxes is None else len(boxes)
    annotated_image = results[0].plot() if show_preview and results else image
    return person_count, annotated_image


def run() -> None:
    # Load .env from the project directory when running locally.  Docker also
    # injects the same values with docker-compose's env_file setting.
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    settings = Settings.from_env()
    _configure_logging(settings.log_level)

    LOGGER.info("載入 YOLO 模型: %s", settings.model_path)
    model = YOLO(settings.model_path)

    window = DetectionWindow(settings.sample_count, settings.detection_min_ratio)
    # Keep image samples aligned with the detection window.  A positive sample
    # stores the full camera frame while a negative sample is represented by
    # None, so the alert email contains every frame that found a person.
    captured_images: deque[object | None] = deque(maxlen=settings.sample_count)
    gate = AlertGate(settings.alert_cooldown_seconds)
    notifier = Notifier(settings)
    stop_event = threading.Event()

    def request_stop(signum: int, _frame: object) -> None:
        LOGGER.info("收到停止訊號 (%s)，正在關閉。", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_stop)

    if settings.camera_source == "webcam":
        source: str | int = settings.webcam_index
        LOGGER.info(
            "使用本機鏡頭：index=%d, resolution=%dx%d, fps=%s。",
            settings.webcam_index,
            settings.webcam_width,
            settings.webcam_height,
            settings.webcam_fps or "default",
        )
    else:
        source = settings.build_rtsp_url()
        # Do not log source: it may contain the RTSP password.
        LOGGER.info("使用 RTSP 影像來源。")

    reader = LatestFrameReader(
        source=source,
        reconnect_delay_seconds=settings.rtsp_reconnect_delay_seconds,
        open_timeout_seconds=settings.rtsp_open_timeout_seconds,
        transport=settings.rtsp_transport,
        frame_width=(settings.webcam_width if settings.camera_source == "webcam" else None),
        frame_height=(settings.webcam_height if settings.camera_source == "webcam" else None),
        frame_fps=(settings.webcam_fps if settings.camera_source == "webcam" else 0.0),
    )
    reader.start()

    preview_enabled = settings.show_preview
    if preview_enabled:
        LOGGER.info("即時預覽已啟用；按 q 或 Esc 可停止程式。")

    LOGGER.info(
        "啟動監測：每 %.1f 秒取樣；視窗 %.1f 秒（%d 次）；至少 %d 次有人才告警。",
        settings.sample_interval_seconds,
        settings.detection_window_seconds,
        settings.sample_count,
        settings.minimum_detections,
    )
    LOGGER.info(
        "目前只輸出到 CMD；EMAIL_ENABLED=%s。",
        settings.email_enabled,
    )

    next_sample_at = time.monotonic()
    try:
        while not stop_event.is_set():
            now = time.monotonic()
            wait_seconds = next_sample_at - now
            if wait_seconds > 0:
                stop_event.wait(wait_seconds)
                continue

            # Keep the schedule roughly aligned to the requested interval. If
            # inference takes too long, skip missed ticks instead of running a
            # backlog of stale frames.
            next_sample_at += settings.sample_interval_seconds
            if next_sample_at <= now:
                next_sample_at = now + settings.sample_interval_seconds

            latest = reader.latest()
            person_count = 0
            frame_age = None
            preview_image = latest.image if latest is not None else None
            if latest is not None:
                frame_age = time.monotonic() - latest.received_at
                if frame_age <= settings.max_frame_age_seconds:
                    try:
                        person_count, preview_image = _count_people(
                            model,
                            latest.image,
                            settings.person_confidence,
                            settings.yolo_device,
                            settings.yolo_image_size,
                            show_preview=preview_enabled,
                        )
                    except Exception:
                        # A single bad frame should not terminate a long-running
                        # service. Count it as a negative sample and continue.
                        LOGGER.exception("YOLO 推論失敗，本次取樣視為沒有人。")
                else:
                    LOGGER.warning(
                        "目前影像已過期 %.1f 秒，本次取樣視為沒有人。",
                        frame_age,
                    )
            else:
                LOGGER.warning("尚未取得 RTSP 影像，本次取樣視為沒有人。")

            stats = window.add(person_count > 0)
            captured_images.append(
                latest.image.copy()
                if person_count > 0 and latest is not None
                else None
            )
            LOGGER.info(
                "取樣：person_count=%d, window=%d/%d, ratio=%.1f%%",
                person_count,
                stats.positive_samples,
                stats.window_size,
                stats.ratio * 100,
            )

            if preview_enabled and preview_image is not None:
                try:
                    cv2.putText(
                        preview_image,
                        f"people={person_count} | samples={stats.positive_samples}/{stats.window_size}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0) if person_count else (0, 0, 255),
                        2,
                    )
                    cv2.imshow("YOLO_CAM preview", preview_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in {ord("q"), 27}:
                        stop_event.set()
                except cv2.error:
                    LOGGER.exception(
                        "無法顯示預覽視窗；請安裝非 headless OpenCV，或將 SHOW_PREVIEW=false。"
                    )
                    preview_enabled = False

            if gate.should_alert(stats, now):
                try:
                    notifier.notify(
                        stats,
                        [image for image in captured_images if image is not None],
                    )
                except Exception:
                    # Keep the camera monitor alive if a future SMTP server is
                    # temporarily unavailable or misconfigured.
                    LOGGER.exception("告警通知失敗。")
    finally:
        reader.stop()
        if settings.show_preview:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        LOGGER.info("監測器已停止。")


if __name__ == "__main__":
    # The web dashboard owns camera readers and the one-second inference
    # schedule.  The legacy ``run`` function remains available for scripts
    # that explicitly import it.
    from .dashboard import run_dashboard

    run_dashboard()

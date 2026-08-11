"""Application settings loaded from environment variables.

The project intentionally keeps all camera credentials and tunable values in
environment variables so that the same image can be used locally and in
Docker.  See ``.env.example`` for the available settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from os import getenv
from urllib.parse import quote


def _get_str(name: str, default: str = "") -> str:
    """Read a string setting and remove accidental surrounding whitespace."""

    return getenv(name, default).strip()


def _get_int(name: str, default: int) -> int:
    value = _get_str(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"環境變數 {name} 必須是整數，目前是: {value!r}") from exc


def _get_float(name: str, default: float) -> float:
    value = _get_str(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"環境變數 {name} 必須是數字，目前是: {value!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    value = _get_str(name, str(default)).lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"環境變數 {name} 必須是 true/false，目前是: {value!r}"
    )


@dataclass(frozen=True)
class Settings:
    """Validated runtime settings."""

    # Camera source.  ``webcam`` is useful for local development; ``rtsp`` is
    # the normal production mode.
    camera_source: str
    webcam_index: int
    webcam_width: int
    webcam_height: int
    webcam_fps: float
    show_preview: bool

    # RTSP camera connection.
    rtsp_url: str
    rtsp_scheme: str
    rtsp_host: str
    rtsp_port: int
    rtsp_username: str
    rtsp_password: str
    rtsp_path: str
    rtsp_transport: str
    rtsp_open_timeout_seconds: float
    rtsp_reconnect_delay_seconds: float

    # YOLO inference.
    model_path: str
    yolo_device: str
    person_confidence: float
    yolo_image_size: int

    # Sampling and alert rule.
    sample_interval_seconds: float
    detection_window_seconds: float
    detection_min_ratio: float
    alert_cooldown_seconds: float
    max_frame_age_seconds: float

    # Notification.  Email is deliberately disabled by default for this first
    # version; alerts are always printed to the command line.
    email_enabled: bool
    smtp_host: str
    smtp_port: int
    smtp_username: str
    smtp_password: str
    smtp_use_tls: bool
    email_from: str
    email_to: tuple[str, ...]
    email_subject: str

    # Web control room.
    web_host: str
    web_port: int
    camera_config_path: str
    roi_dwell_seconds: float

    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        camera_source = _get_str("CAMERA_SOURCE", "rtsp").lower()
        if camera_source not in {"rtsp", "webcam"}:
            raise ValueError("CAMERA_SOURCE 必須是 rtsp 或 webcam")

        rtsp_host = _get_str("RTSP_HOST")
        rtsp_url = _get_str("RTSP_URL")
        if camera_source == "rtsp" and not rtsp_url and not rtsp_host:
            raise ValueError("請至少設定 RTSP_URL 或 RTSP_HOST")

        webcam_index = _get_int("WEBCAM_INDEX", 0)
        webcam_width = _get_int("WEBCAM_WIDTH", 1280)
        webcam_height = _get_int("WEBCAM_HEIGHT", 720)
        webcam_fps = _get_float("WEBCAM_FPS", 0.0)
        if webcam_index < 0:
            raise ValueError("WEBCAM_INDEX 不可小於 0")
        if webcam_width <= 0 or webcam_height <= 0:
            raise ValueError("WEBCAM_WIDTH 與 WEBCAM_HEIGHT 必須大於 0")
        if webcam_fps < 0:
            raise ValueError("WEBCAM_FPS 不可小於 0；0 代表使用相機預設值")

        sample_interval = _get_float("SAMPLE_INTERVAL_SECONDS", 1.0)
        window_seconds = _get_float("DETECTION_WINDOW_SECONDS", 15.0)
        min_ratio = _get_float("DETECTION_MIN_RATIO", 2 / 3)

        if sample_interval <= 0:
            raise ValueError("SAMPLE_INTERVAL_SECONDS 必須大於 0")
        if window_seconds <= 0:
            raise ValueError("DETECTION_WINDOW_SECONDS 必須大於 0")
        if not 0 < min_ratio <= 1:
            raise ValueError("DETECTION_MIN_RATIO 必須大於 0 且小於等於 1")

        person_confidence = _get_float("PERSON_CONFIDENCE", 0.5)
        if not 0 <= person_confidence <= 1:
            raise ValueError("PERSON_CONFIDENCE 必須介於 0 與 1 之間")

        yolo_image_size = _get_int("YOLO_IMAGE_SIZE", 640)
        if yolo_image_size < 32 or yolo_image_size % 32:
            raise ValueError("YOLO_IMAGE_SIZE must be a multiple of 32 and at least 32")

        email_to = tuple(
            address.strip()
            for address in _get_str("EMAIL_TO").split(",")
            if address.strip()
        )

        return cls(
            camera_source=camera_source,
            webcam_index=webcam_index,
            webcam_width=webcam_width,
            webcam_height=webcam_height,
            webcam_fps=webcam_fps,
            show_preview=_get_bool("SHOW_PREVIEW", camera_source == "webcam"),
            rtsp_url=rtsp_url,
            rtsp_scheme=_get_str("RTSP_SCHEME", "rtsp"),
            rtsp_host=rtsp_host,
            rtsp_port=_get_int("RTSP_PORT", 554),
            rtsp_username=_get_str("RTSP_USERNAME"),
            rtsp_password=_get_str("RTSP_PASSWORD"),
            rtsp_path=_get_str("RTSP_PATH", "/Streaming/Channels/101"),
            rtsp_transport=_get_str("RTSP_TRANSPORT", "tcp"),
            rtsp_open_timeout_seconds=_get_float(
                "RTSP_OPEN_TIMEOUT_SECONDS", 10.0
            ),
            rtsp_reconnect_delay_seconds=_get_float(
                "RTSP_RECONNECT_DELAY_SECONDS", 5.0
            ),
            model_path=_get_str("MODEL_PATH", "yolo26n.pt"),
            yolo_device=_get_str("YOLO_DEVICE", "cpu"),
            person_confidence=person_confidence,
            yolo_image_size=yolo_image_size,
            sample_interval_seconds=sample_interval,
            detection_window_seconds=window_seconds,
            detection_min_ratio=min_ratio,
            alert_cooldown_seconds=_get_float("ALERT_COOLDOWN_SECONDS", 300.0),
            max_frame_age_seconds=_get_float("MAX_FRAME_AGE_SECONDS", 5.0),
            email_enabled=_get_bool("EMAIL_ENABLED", False),
            smtp_host=_get_str("SMTP_HOST"),
            smtp_port=_get_int("SMTP_PORT", 587),
            smtp_username=_get_str("SMTP_USERNAME"),
            smtp_password=_get_str("SMTP_PASSWORD"),
            smtp_use_tls=_get_bool("SMTP_USE_TLS", True),
            email_from=_get_str("EMAIL_FROM"),
            email_to=email_to,
            email_subject=_get_str(
                "EMAIL_SUBJECT", "YOLO CAM person detected"
            ),
            web_host=_get_str("WEB_HOST", "0.0.0.0"),
            web_port=_get_int("WEB_PORT", 8080),
            camera_config_path=_get_str("CAMERA_CONFIG_PATH", "data/cameras.json"),
            roi_dwell_seconds=_get_float("ROI_DWELL_SECONDS", 10.0),
            log_level=_get_str("LOG_LEVEL", "INFO").upper(),
        )

    @property
    def sample_count(self) -> int:
        """Number of samples in one detection window.

        ``ceil`` ensures that a window is never shorter than the configured
        duration when the sampling interval is changed from one second.
        """

        return max(1, ceil(self.detection_window_seconds / self.sample_interval_seconds))

    @property
    def minimum_detections(self) -> int:
        """Minimum positive samples required to trigger an alert."""

        return max(1, ceil(self.sample_count * self.detection_min_ratio))

    def build_rtsp_url(self) -> str:
        """Build the RTSP URL from the individual .env fields.

        ``RTSP_URL`` is supported for cameras with a vendor-specific URL.  If
        it is empty, the normal ``scheme://user:password@host:port/path`` form
        is generated and credentials are URL-encoded safely.
        """

        if self.rtsp_url:
            return self.rtsp_url

        credentials = ""
        if self.rtsp_username:
            credentials = quote(self.rtsp_username, safe="")
            if self.rtsp_password:
                credentials += f":{quote(self.rtsp_password, safe='')}"
            credentials += "@"

        path = self.rtsp_path or "/"
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.rtsp_scheme}://{credentials}{self.rtsp_host}:{self.rtsp_port}{path}"

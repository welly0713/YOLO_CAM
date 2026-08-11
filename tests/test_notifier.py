from email.message import EmailMessage

import numpy as np

from app.config import Settings
from app.detection import WindowStats
from app.notifier import Notifier


class FakeSMTP:
    sent: EmailMessage | None = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def login(self, *args):
        pass

    def send_message(self, message: EmailMessage):
        type(self).sent = message


def _settings() -> Settings:
    return Settings(
        camera_source="webcam", webcam_index=0, webcam_width=1, webcam_height=1,
        webcam_fps=0, show_preview=False, rtsp_url="", rtsp_scheme="rtsp",
        rtsp_host="", rtsp_port=554, rtsp_username="", rtsp_password="",
        rtsp_path="/", rtsp_transport="tcp", rtsp_open_timeout_seconds=1,
        rtsp_reconnect_delay_seconds=1, model_path="model.pt", yolo_device="cpu",
        person_confidence=0.5, yolo_image_size=640, sample_interval_seconds=1,
        detection_window_seconds=1, detection_min_ratio=1,
        alert_cooldown_seconds=1, max_frame_age_seconds=1, email_enabled=True,
        smtp_host="smtp.test", smtp_port=25, smtp_username="", smtp_password="",
        smtp_use_tls=False, email_from="from@test", email_to=("to@test",),
        email_subject="Person detected", web_host="127.0.0.1", web_port=8080,
        camera_config_path="data/cameras.json", roi_dwell_seconds=10,
        log_level="INFO",
    )


def test_email_attaches_every_person_frame(monkeypatch):
    monkeypatch.setattr("app.notifier.smtplib.SMTP", FakeSMTP)
    FakeSMTP.sent = None
    stats = WindowStats(samples=2, positive_samples=2, required_samples=2, window_size=2)

    Notifier(_settings()).notify(stats, [np.zeros((4, 4, 3), dtype=np.uint8)] * 2)

    assert FakeSMTP.sent is not None
    attachments = list(FakeSMTP.sent.iter_attachments())
    assert len(attachments) == 2
    assert all(item.get_content_type() == "image/jpeg" for item in attachments)

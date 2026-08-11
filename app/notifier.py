"""Alert output.

The console output is the default for the first version.  SMTP support is
included and can be enabled later by setting ``EMAIL_ENABLED=true`` in .env.
"""

from __future__ import annotations

import logging
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from typing import Sequence

import cv2

from .config import Settings
from .detection import WindowStats


LOGGER = logging.getLogger(__name__)


class Notifier:
    """Print every alert and optionally send the same alert through SMTP."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def notify(self, stats: WindowStats, images: Sequence[object] = ()) -> None:
        """Send an alert and attach every sampled frame containing a person.

        Whole frames are used instead of crops so every person in the image,
        together with useful context, is retained in the attachment.
        """
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        message = (
            f"[ALERT] {now} 偵測到有人："
            f"最近 {stats.window_size} 次取樣中 {stats.positive_samples}/"
            f"{stats.window_size} 次有人（門檻 {stats.required_samples} 次）。"
        )

        # This is the requested behavior for the initial version.
        print(message, flush=True)

        if self.settings.email_enabled:
            self._send_email(message, images)

    def _send_email(self, body: str, images: Sequence[object]) -> None:
        settings = self.settings
        missing = []
        if not settings.smtp_host:
            missing.append("SMTP_HOST")
        if not settings.email_from:
            missing.append("EMAIL_FROM")
        if not settings.email_to:
            missing.append("EMAIL_TO")
        if missing:
            raise ValueError(
                "EMAIL_ENABLED=true 但缺少必要設定: " + ", ".join(missing)
            )

        email = EmailMessage()
        email["Subject"] = settings.email_subject
        email["From"] = settings.email_from
        email["To"] = ", ".join(settings.email_to)
        email.set_content(body)

        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        attachment_count = 0
        for index, image in enumerate(images, start=1):
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                LOGGER.warning("Skipping a person frame that could not be JPEG encoded")
                continue
            email.add_attachment(
                encoded.tobytes(),
                maintype="image",
                subtype="jpeg",
                filename=f"person-{timestamp}-{index:02d}.jpg",
            )
            attachment_count += 1

        LOGGER.info("Sending person alert email with %d image attachment(s)", attachment_count)

        LOGGER.info("正在透過 SMTP 寄送告警郵件。")
        if settings.smtp_use_tls:
            with smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=15,
            ) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                if settings.smtp_username:
                    server.login(settings.smtp_username, settings.smtp_password)
                server.send_message(email)
        else:
            with smtplib.SMTP(
                settings.smtp_host,
                settings.smtp_port,
                timeout=15,
            ) as server:
                if settings.smtp_username:
                    server.login(settings.smtp_username, settings.smtp_password)
                server.send_message(email)

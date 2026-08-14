from dataclasses import replace
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.config import Settings
from app.dashboard import CameraConfigStore, _alert_timestamp, create_app


class FakeYOLO:
    def __init__(self, *args, **kwargs):
        pass


def test_alert_timestamp_uses_taipei_time():
    recorded = datetime.strptime(_alert_timestamp(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Taipei"))
    expected = datetime.now(ZoneInfo("Asia/Taipei"))
    assert abs((recorded - expected).total_seconds()) < 2


def test_dashboard_has_three_live_slots_and_four_camera_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE", "webcam")
    monkeypatch.setenv("DASHBOARD_SETTINGS_PASSWORD", "test-password")
    settings = replace(Settings.from_env(), camera_config_path=str(tmp_path / "cameras.json"))
    with patch("app.dashboard.YOLO", FakeYOLO):
        app = create_app(settings)
        client = app.test_client()
        home = client.get("/")
        assert home.status_code == 200
        assert b"cam-1" in home.data and b"cam-3" in home.data
        assert "ROI 內停留".encode() in home.data
        assert b"sawtooth" in home.data
        assert "確認關閉本次警報".encode() in home.data
        assert b"o.stop(t+30)" in home.data
        assert client.get("/settings").status_code == 302
        assert client.get("/api/config").status_code == 401
        login = client.post("/settings/login", data={"password": "test-password"})
        assert login.status_code == 302
        assert client.get(login.location).status_code == 200
        assert client.get("/settings").status_code == 302
        login = client.post("/settings/login", data={"password": "test-password"})
        assert client.get(login.location).status_code == 200
        config = client.get("/api/config").get_json()
        assert len(config["cameras"]) == 4
        assert config["cameras"][0]["alert_enabled"] is True
        config["cameras"][0]["roi"] = [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]
        assert client.post("/api/config", json=config).status_code == 200

    persisted = CameraConfigStore(str(tmp_path / "cameras.json"), settings).get()
    assert len(persisted["cameras"][0]["roi"]) == 3

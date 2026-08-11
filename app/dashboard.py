"""Browser-based control room for four IP cameras and ROI dwell alerts."""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, render_template_string, request
from ultralytics import YOLO

from .camera import LatestFrameReader
from .config import Settings


LOGGER = logging.getLogger(__name__)
CAMERA_IDS = tuple(f"cam-{index}" for index in range(1, 5))
LOCAL_TIMEZONE = ZoneInfo("Asia/Taipei")


def _alert_timestamp() -> str:
    """Return dashboard event timestamps in the control room's local timezone."""
    return datetime.now(LOCAL_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _default_camera(camera_id: str, index: int, settings: Settings) -> dict[str, Any]:
    url = settings.build_rtsp_url() if settings.camera_source == "rtsp" and index == 1 else ""
    return {
        "id": camera_id,
        "name": f"IP Camera {index}",
        "enabled": bool(url),
        "rtsp_url": url,
        "roi": [],
        "dwell_seconds": settings.roi_dwell_seconds,
    }


class CameraConfigStore:
    """Persistent editable configuration for four camera/ROI definitions."""

    def __init__(self, path: str, settings: Settings) -> None:
        self.path = Path(path)
        self.settings = settings
        self._lock = threading.Lock()
        self._data = self._read_or_seed()

    def _read_or_seed(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return self._normalise(data)
        except FileNotFoundError:
            data = {"cameras": [_default_camera(cid, i, self.settings) for i, cid in enumerate(CAMERA_IDS, 1)]}
            self._write(data)
            return data
        except (OSError, json.JSONDecodeError):
            LOGGER.exception("Camera config is invalid; starting with safe defaults")
            return {"cameras": [_default_camera(cid, i, self.settings) for i, cid in enumerate(CAMERA_IDS, 1)]}

    def _normalise(self, value: Any) -> dict[str, Any]:
        incoming = value.get("cameras", []) if isinstance(value, dict) else []
        by_id = {item.get("id"): item for item in incoming if isinstance(item, dict)}
        cameras: list[dict[str, Any]] = []
        for index, camera_id in enumerate(CAMERA_IDS, 1):
            item = _default_camera(camera_id, index, self.settings)
            item.update(by_id.get(camera_id, {}))
            item["id"] = camera_id
            item["name"] = str(item["name"])[:80] or f"IP Camera {index}"
            item["enabled"] = bool(item["enabled"])
            item["rtsp_url"] = str(item["rtsp_url"]).strip()
            item["dwell_seconds"] = max(1.0, min(float(item["dwell_seconds"]), 3600.0))
            roi = item.get("roi", [])
            item["roi"] = [
                [round(float(point[0]), 6), round(float(point[1]), 6)]
                for point in roi
                if isinstance(point, list) and len(point) == 2
                and 0 <= float(point[0]) <= 1 and 0 <= float(point[1]) <= 1
            ][:20]
            cameras.append(item)
        return {"cameras": cameras}

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)

    def save(self, value: Any) -> dict[str, Any]:
        data = self._normalise(value)
        with self._lock:
            self._write(data)
            self._data = data
            return copy.deepcopy(data)


@dataclass
class Track:
    point: tuple[float, float]
    entered_at: float
    last_seen: float
    alerted: bool = False


@dataclass
class CameraState:
    online: bool = False
    last_frame_at: float | None = None
    last_detected_at: float | None = None
    detections: list[tuple[int, int, int, int]] = field(default_factory=list)
    tracks: dict[int, Track] = field(default_factory=dict)
    next_track_id: int = 1
    alert_sequence: int = 0
    last_alert: dict[str, Any] | None = None


class CameraWorker:
    """Continuously streams one camera, while performing YOLO only once/sec."""

    def __init__(self, config: dict[str, Any], settings: Settings, model: YOLO, model_lock: threading.Lock) -> None:
        self.config = config
        self.settings = settings
        self.model = model
        self.model_lock = model_lock
        self.state = CameraState()
        self.state_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.reader: LatestFrameReader | None = None
        self.thread: threading.Thread | None = None

    @property
    def id(self) -> str:
        return self.config["id"]

    def start(self) -> None:
        if not self.config["enabled"] or not self.config["rtsp_url"]:
            return
        self.reader = LatestFrameReader(
            source=self.config["rtsp_url"],
            reconnect_delay_seconds=self.settings.rtsp_reconnect_delay_seconds,
            open_timeout_seconds=self.settings.rtsp_open_timeout_seconds,
            transport=self.settings.rtsp_transport,
        )
        self.reader.start()
        self.thread = threading.Thread(target=self._detect_loop, name=f"detector-{self.id}", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.reader:
            self.reader.stop()
        if self.thread:
            self.thread.join(timeout=3)

    def _inside_roi(self, point: tuple[float, float], shape: tuple[int, ...]) -> bool:
        roi = self.config["roi"]
        if len(roi) < 3:
            return False
        height, width = shape[:2]
        polygon = np.array([(x * width, y * height) for x, y in roi], dtype=np.float32)
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def _update_tracks(self, points: list[tuple[float, float]], now: float) -> None:
        """Assign detections to nearby tracks; IDs are only valid for this event."""
        with self.state_lock:
            available = set(self.state.tracks)
            for point in points:
                match = min(
                    available,
                    key=lambda key: (self.state.tracks[key].point[0] - point[0]) ** 2
                    + (self.state.tracks[key].point[1] - point[1]) ** 2,
                    default=None,
                )
                if match is not None:
                    distance = np.hypot(self.state.tracks[match].point[0] - point[0], self.state.tracks[match].point[1] - point[1])
                    if distance <= 150:
                        track = self.state.tracks[match]
                        track.point, track.last_seen = point, now
                        available.remove(match)
                        continue
                track_id = self.state.next_track_id
                self.state.next_track_id += 1
                self.state.tracks[track_id] = Track(point=point, entered_at=now, last_seen=now)

            expired = [key for key, track in self.state.tracks.items() if now - track.last_seen > 3]
            for key in expired:
                del self.state.tracks[key]

            for track_id, track in self.state.tracks.items():
                if not track.alerted and now - track.entered_at >= self.config["dwell_seconds"]:
                    track.alerted = True
                    self.state.alert_sequence += 1
                    self.state.last_alert = {
                        "track_id": track_id,
                        "at": _alert_timestamp(),
                        "dwell_seconds": round(now - track.entered_at, 1),
                    }
                    LOGGER.warning("%s: person stayed in ROI for %.1f seconds", self.config["name"], now - track.entered_at)

    def _detect_loop(self) -> None:
        while not self.stop_event.is_set():
            started = time.monotonic()
            latest = self.reader.latest() if self.reader else None
            if latest:
                with self.state_lock:
                    self.state.online = True
                    self.state.last_frame_at = latest.received_at
                try:
                    with self.model_lock:
                        results = self.model.predict(latest.image, conf=self.settings.person_confidence, classes=[0], imgsz=self.settings.yolo_image_size, device=self.settings.yolo_device, verbose=False)
                    boxes = results[0].boxes if results else None
                    detections: list[tuple[int, int, int, int]] = []
                    points: list[tuple[float, float]] = []
                    if boxes is not None:
                        for x1, y1, x2, y2 in boxes.xyxy.cpu().tolist():
                            foot = ((x1 + x2) / 2, y2)
                            if self._inside_roi(foot, latest.image.shape):
                                detections.append((int(x1), int(y1), int(x2), int(y2)))
                                points.append(foot)
                    with self.state_lock:
                        self.state.detections = detections
                        if detections:
                            self.state.last_detected_at = time.monotonic()
                    self._update_tracks(points, time.monotonic())
                except Exception:
                    LOGGER.exception("%s: YOLO inference failed", self.config["name"])
            else:
                with self.state_lock:
                    self.state.online = False
            self.stop_event.wait(max(0.0, 1.0 - (time.monotonic() - started)))

    def status(self) -> dict[str, Any]:
        with self.state_lock:
            active = [
                {"id": track_id, "seconds": round(time.monotonic() - track.entered_at, 1), "alerted": track.alerted}
                for track_id, track in self.state.tracks.items()
            ]
            return {
                "id": self.id, "name": self.config["name"], "enabled": self.config["enabled"],
                "online": self.state.online, "roi_configured": len(self.config["roi"]) >= 3,
                "detections": len(self.state.detections), "tracks": active,
                "alert_sequence": self.state.alert_sequence, "last_alert": self.state.last_alert,
            }

    def jpeg(self) -> bytes | None:
        latest = self.reader.latest() if self.reader else None
        if latest is None:
            return None
        image = latest.image
        height, width = image.shape[:2]
        roi = self.config["roi"]
        if len(roi) >= 3:
            polygon = np.array([(int(x * width), int(y * height)) for x, y in roi], dtype=np.int32)
            cv2.polylines(image, [polygon], True, (0, 215, 255), 4)
        with self.state_lock:
            boxes = list(self.state.detections)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return encoded.tobytes() if ok else None


class MonitorService:
    def __init__(self, settings: Settings, store: CameraConfigStore) -> None:
        self.settings, self.store = settings, store
        self.model = YOLO(settings.model_path)
        self.model_lock = threading.Lock()
        self.workers: dict[str, CameraWorker] = {}
        self.apply(store.get())

    def apply(self, config: dict[str, Any]) -> None:
        for worker in self.workers.values():
            worker.stop()
        self.workers = {
            item["id"]: CameraWorker(item, self.settings, self.model, self.model_lock)
            for item in config["cameras"]
        }
        for worker in self.workers.values():
            worker.start()

    def status(self) -> list[dict[str, Any]]:
        return [self.workers[camera_id].status() for camera_id in CAMERA_IDS]

    def stream(self, camera_id: str):
        worker = self.workers.get(camera_id)
        if worker is None:
            abort(404)
        while True:
            image = worker.jpeg()
            if image:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + image + b"\r\n"
            else:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera offline", (190, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
                ok, encoded = cv2.imencode(".jpg", blank)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            time.sleep(0.2)


HOME = """<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>YOLO CAM 中控台</title><style>
body{margin:0;background:#111827;color:#e5e7eb;font:16px system-ui}.top{height:64px;display:flex;align-items:center;gap:24px;padding:0 24px;background:#1f2937;border-bottom:1px solid #374151}.top h1{font-size:20px;margin:0}.pill{padding:6px 10px;background:#374151;border-radius:999px}.right{margin-left:auto}.right a{color:#93c5fd}.grid{padding:18px;display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{background:#1f2937;border:1px solid #374151;border-radius:10px;overflow:hidden}.card h2{font-size:16px;margin:0;padding:10px 12px;display:flex;justify-content:space-between}.card img{width:100%;aspect-ratio:16/9;object-fit:cover;background:#000;display:block}.info{padding:10px 12px;color:#cbd5e1}.ok{color:#4ade80}.bad{color:#f87171}.alert{outline:4px solid #ef4444}.empty{display:flex;align-items:center;justify-content:center;min-height:200px;color:#64748b}</style></head><body>
<header class='top'><h1>YOLO CAM 中控台</h1><span id='summary' class='pill'>載入中…</span><span class='right'><button id='sound'>啟用告警聲</button>　<a href='/settings'>設定攝影機與 ROI</a></span></header><main class='grid' id='grid'></main><script>
let last={},audioContext; const grid=document.querySelector('#grid');document.querySelector('#sound').onclick=()=>{audioContext=new AudioContext();audioContext.resume();document.querySelector('#sound').textContent='告警聲已啟用'};function beep(){if(!audioContext)return;for(let i=0;i<3;i++){let oscillator=audioContext.createOscillator(),gain=audioContext.createGain(),at=audioContext.currentTime+i*.22;oscillator.frequency.value=880;gain.gain.setValueAtTime(.15,at);gain.gain.exponentialRampToValueAtTime(.001,at+.18);oscillator.connect(gain).connect(audioContext.destination);oscillator.start(at);oscillator.stop(at+.18)}}
function card(c){return `<section class='card' id='${c.id}'><h2>${c.name}<span class='state'></span></h2><img src='/stream/${c.id}'><div class='info'></div></section>`} grid.innerHTML=['cam-1','cam-2','cam-3'].map(id=>card({id,name:id})).join('');
async function refresh(){let list=await (await fetch('/api/status')).json(), online=0; for(const c of list){let e=document.querySelector('#'+c.id);if(!e)continue;let status=e.querySelector('.state'),info=e.querySelector('.info');status.innerHTML=c.online?'<b class="ok">● 連線中</b>':'<b class="bad">● 離線</b>';info.textContent=!c.enabled?'尚未啟用':!c.roi_configured?'請在設定頁設定 ROI':c.tracks.length?`ROI 內 ${c.detections} 人｜逗留 ${c.tracks.map(x=>x.seconds+' 秒').join(', ')}`:'ROI 內無人';e.classList.toggle('alert',c.tracks.some(x=>x.alerted));if(c.online)online++;if((last[c.id]||0)<c.alert_sequence){beep();last[c.id]=c.alert_sequence}}document.querySelector('#summary').textContent=`${online}/4 台攝影機連線中`};refresh();setInterval(refresh,1000);</script></body></html>"""


HOME = """<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>汙水排放口即時警戒系統</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b0f12;color:#c6d0d3;font:12px ui-monospace,SFMono-Regular,Consolas,monospace;min-width:1080px}.top{height:39px;border-bottom:1px solid #273036;display:flex;align-items:center;padding:0 13px;color:#849197}.brand{font-weight:700;color:#cbd5d8;font-size:13px}.brand b{color:#eef5f5}.timestamp{margin-left:auto;color:#718087;font-size:10px}.sys{margin-left:13px;padding:6px 10px;border:1px solid #167b75;background:#073f3c;color:#5ce3d6;border-radius:2px;font-size:10px}.dot{color:#19dbc8}.shell{display:grid;grid-template-columns:minmax(760px,1fr) 226px;gap:10px;padding:10px;min-height:calc(100vh - 65px)}.cameras{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));grid-template-rows:repeat(2,minmax(210px,1fr));gap:10px}.feed{position:relative;background:#020405;border:1px solid #253038;border-radius:4px;overflow:hidden;min-height:0}.feed.alert{border-color:#d64b4b;box-shadow:0 0 0 1px #742b2b}.feed img{height:100%;width:100%;display:block;object-fit:cover;opacity:.88}.camtime{position:absolute;top:9px;left:10px;color:#b7c5c7;text-shadow:0 1px 3px #000;font-size:10px}.camname{position:absolute;bottom:10px;left:10px;color:#e6eeee;text-shadow:0 1px 4px #000;font-weight:bold}.live{position:absolute;bottom:10px;right:10px;color:#1be0cb;font-size:10px}.live.off{color:#647277}.side{display:flex;flex-direction:column;gap:10px}.panel{border:1px solid #263139;border-radius:4px;background:#13191d;padding:11px}.panel h2{font-size:12px;color:#9ba9ad;margin:0 0 11px}.metric{display:flex;justify-content:space-between;border-bottom:1px solid #222c31;padding:7px 0}.metric:last-of-type{border:0}.value{color:#15bfb2;font-weight:bold}.event{font-size:10px;border-left:2px solid #1caaa0;background:#192124;padding:8px;margin-top:6px;color:#aebabc}.event.alert{border-left-color:#e35b5b}.empty{color:#66767d;padding:8px 0;font-size:10px}.sound{width:100%;margin-top:12px;border:1px dashed #67767a;background:transparent;color:#9caaad;padding:8px;cursor:pointer;font:inherit}.sound.on{border-color:#139f96;color:#35d5c9}.footer{height:26px;border-top:1px solid #20292e;padding:7px 12px;color:#59676c;font-size:10px;white-space:nowrap;overflow:hidden}</style></head><body>
<header class='top'><span class='brand'><b>汙水排放口 即時警戒系統</b></span><span class='timestamp' id='clock'></span><button class='sys' id='sound'><span class='dot'>●</span> 啟用告警聲</button></header><main class='shell'><section class='cameras' id='cameras'></section><aside class='side'><section class='panel'><h2>今日概況</h2><div class='metric'><span>監控攝影機</span><span class='value' id='online'>0 / 4</span></div><div class='metric'><span>YOLO 偵測服務</span><span class='value'>運作中</span></div><div class='metric'><span>今日觸發次數</span><span class='value' id='event-count'>0</span></div><div class='metric'><span>上次警報</span><span class='value' id='last-alert'>尚無資料</span></div><button class='sound' id='settings' onclick="location.href='/settings'">▶ 攝影機／ROI 設定</button></section><section class='panel'><h2>事件紀錄</h2><div id='events'><div class='empty'>尚未有告警事件</div></div></section></aside></main><footer class='footer'>偵測模式：人員逗留於取水口 ROI 區域才告警。每台攝影機每秒僅執行一次 YOLO 推論；即時畫面持續串流。</footer><script>
let last={},events=[],audioContext;const area=document.querySelector('#cameras');function card(id){return `<article class='feed' id='${id}'><img src='/stream/${id}'><span class='camtime'></span><span class='camname'>${id}</span><span class='live off'>● OFFLINE</span></article>`}area.innerHTML=['cam-1','cam-2','cam-3','cam-4'].map(card).join('');function beep(){if(!audioContext)return;for(let i=0;i<3;i++){const o=audioContext.createOscillator(),g=audioContext.createGain(),t=audioContext.currentTime+i*.22;o.frequency.value=880;g.gain.setValueAtTime(.14,t);g.gain.exponentialRampToValueAtTime(.001,t+.18);o.connect(g).connect(audioContext.destination);o.start(t);o.stop(t+.18)}}document.querySelector('#sound').onclick=()=>{audioContext=new AudioContext();audioContext.resume();let b=document.querySelector('#sound');b.innerHTML='<span class="dot">●</span> 系統正常監控中';b.classList.add('on')};function drawEvents(){let target=document.querySelector('#events');target.innerHTML=events.length?events.map(e=>`<div class='event alert'><b>${e.name}</b><br>${e.at}　逗留 ${e.dwell_seconds} 秒</div>`).join(''):'<div class="empty">尚未有告警事件</div>'}function tick(){document.querySelector('#clock').textContent=new Date().toLocaleString('zh-TW',{hour12:false})}async function refresh(){const list=await (await fetch('/api/status')).json();let online=0,total=0,newest=null;for(const c of list){const e=document.querySelector('#'+c.id);e.querySelector('.camname').textContent=c.name;e.querySelector('.camtime').textContent=new Date().toLocaleString('sv-SE');const live=e.querySelector('.live');live.textContent=c.online?'● LIVE':'● OFFLINE';live.classList.toggle('off',!c.online);e.classList.toggle('alert',c.tracks.some(x=>x.alerted));if(c.online)online++;total+=c.alert_sequence;if(c.last_alert&&(!newest||c.last_alert.at>newest.at))newest={...c.last_alert,name:c.name};if((last[c.id]||0)<c.alert_sequence){events.unshift({...c.last_alert,name:c.name});events=events.slice(0,8);beep();drawEvents();last[c.id]=c.alert_sequence}}document.querySelector('#online').textContent=`${online} / 4 連線中`;document.querySelector('#event-count').textContent=total;document.querySelector('#last-alert').textContent=newest?`${newest.name} ${newest.at.slice(11)}`:'尚無資料'}tick();setInterval(tick,1000);refresh();setInterval(refresh,1000);</script></body></html>"""


SETTINGS = """<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>YOLO CAM 設定</title><style>body{max-width:1200px;margin:30px auto;background:#111827;color:#e5e7eb;font:16px system-ui;padding:0 20px}a{color:#93c5fd}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.card{background:#1f2937;padding:16px;border-radius:10px}.preview{position:relative}.preview img,.preview canvas{width:100%;aspect-ratio:16/9;position:absolute;left:0;top:0}.preview{aspect-ratio:16/9;background:#000;margin:10px 0}.preview canvas{cursor:crosshair}label{display:block;margin:8px 0}input{width:100%;box-sizing:border-box;padding:8px}button{padding:9px 14px;margin:8px 4px 0 0}#message{position:sticky;top:0;padding:10px;background:#14532d;display:none}</style></head><body><p><a href='/'>← 返回中控台</a></p><h1>攝影機與取水口 ROI 設定</h1><p>在預覽畫面上依序點選取水口範圍的至少三個角點。黃色線框即為告警區域。</p><div id='message'></div><main class='grid' id='grid'></main><button id='save'>儲存並套用設定</button><script>
let config;const grid=document.querySelector('#grid');function draw(c,ctx,can){ctx.clearRect(0,0,can.width,can.height);if(c.roi.length<1)return;ctx.strokeStyle='#facc15';ctx.lineWidth=3;ctx.beginPath();c.roi.forEach((p,i)=>{let x=p[0]*can.width,y=p[1]*can.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});if(c.roi.length>2)ctx.closePath();ctx.stroke();c.roi.forEach(p=>{ctx.fillStyle='#facc15';ctx.fillRect(p[0]*can.width-4,p[1]*can.height-4,8,8)})}function render(){grid.innerHTML='';config.cameras.forEach(c=>{let d=document.createElement('section');d.className='card';d.innerHTML=`<label>名稱<input data-k='name' value='${c.name}'></label><label><input type='checkbox' data-k='enabled' ${c.enabled?'checked':''}> 啟用此攝影機</label><label>RTSP URL<input data-k='rtsp_url' value='${c.rtsp_url}' placeholder='rtsp://帳號:密碼@IP:554/path'></label><label>逗留秒數<input data-k='dwell_seconds' type='number' min='1' value='${c.dwell_seconds}'></label><div class='preview'><img src='/stream/${c.id}'><canvas></canvas></div><button class='clear'>清除 ROI</button>`;grid.append(d);for(const inp of d.querySelectorAll('input'))inp.oninput=()=>{c[inp.dataset.k]=inp.type==='checkbox'?inp.checked:inp.value};let can=d.querySelector('canvas'),ctx=can.getContext('2d');can.width=640;can.height=360;draw(c,ctx,can);can.onclick=e=>{let r=can.getBoundingClientRect();c.roi.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);draw(c,ctx,can)};d.querySelector('.clear').onclick=()=>{c.roi=[];draw(c,ctx,can)}})}async function load(){config=await (await fetch('/api/config')).json();render()}document.querySelector('#save').onclick=async()=>{let r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});let m=document.querySelector('#message');m.textContent=r.ok?'已儲存並重新連線套用設定':'儲存失敗';m.style.display='block'};load();</script></body></html>"""


SETTINGS = """<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><title>攝影機與 ROI 設定</title><style>
*{box-sizing:border-box}body{margin:0;background:#0b0f12;color:#c6d0d3;font:12px ui-monospace,SFMono-Regular,Consolas,monospace;min-width:1080px}.top{height:49px;border-bottom:1px solid #273036;display:flex;align-items:center;padding:0 13px;color:#849197;position:sticky;top:0;background:#0b0f12;z-index:20}.brand{font-weight:700;color:#eef5f5;font-size:13px}.hint{margin-left:18px;color:#647178;font-size:10px}.spacer{margin-left:auto}.back{color:#8aa3a6;text-decoration:none;margin-right:14px}.save{border:1px solid #168f86;background:#07514d;color:#d8fffa;font:inherit;padding:9px 16px;border-radius:3px;cursor:pointer;font-weight:bold}.save:hover{background:#087066}.message{margin-right:13px;color:#54d7c9}.shell{padding:14px;max-width:1400px;margin:auto}.intro{border:1px solid #263139;background:#13191d;border-radius:4px;padding:12px 14px;margin-bottom:12px}.intro h1{margin:0 0 6px;color:#dce5e6;font-size:14px}.intro p{margin:0;color:#76858a;font-size:11px}.grid{display:grid;grid-template-columns:repeat(2,minmax(440px,1fr));gap:12px}.card{background:#13191d;border:1px solid #263139;border-radius:4px;padding:12px}.card:focus-within{border-color:#168f86}.cam-title{color:#ccd7d9;margin:0 0 10px;font-size:12px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:9px}.fields .wide{grid-column:1/-1}label{display:block;color:#849398;font-size:10px}input{display:block;width:100%;margin-top:5px;padding:8px 9px;border-radius:2px;border:1px solid #354249;background:#0b0f12;color:#dbe6e7;font:inherit}input:focus{outline:1px solid #1ba99e}.enabled{display:flex;align-items:center;gap:7px;padding-top:17px;color:#b6c3c5}.enabled input{width:auto;margin:0;accent-color:#12a99d}.preview{position:relative;aspect-ratio:16/9;background:#020405;border:1px solid #273238;margin:12px 0 8px;overflow:hidden}.preview img,.preview canvas{width:100%;height:100%;position:absolute;inset:0}.preview img{object-fit:cover;opacity:.85}.preview canvas{cursor:crosshair}.roi-line{display:flex;justify-content:space-between;align-items:center;color:#718187;font-size:10px}.clear{border:1px solid #48565a;background:transparent;color:#aebdbe;padding:6px 10px;border-radius:2px;cursor:pointer;font:inherit}.clear:hover{border-color:#f0bd52;color:#f6d080}.footer{border-top:1px solid #20292e;padding:8px 13px;color:#59676c;font-size:10px}</style></head><body>
<header class='top'><span class='brand'>汙水排放口 即時警戒系統</span><span class='hint'>攝影機與取水口 ROI 設定</span><span class='spacer'></span><span class='message' id='message'></span><a class='back' href='/'>← 返回中控台</a><button id='save' class='save'>儲存並套用設定</button></header><main class='shell'><section class='intro'><h1>設定四台 IP Camera 與警戒區域</h1><p>填入 RTSP URL，勾選啟用後，在預覽畫面點選至少三個角點圈選 ROI。人員連續逗留達指定秒數才會建立警報。</p></section><section class='grid' id='grid'></section></main><footer class='footer'>儲存後，已啟用的攝影機將短暫重新連線並立即套用新的 ROI 與逗留規則。</footer><script>
let config;const grid=document.querySelector('#grid');function draw(c,ctx,can){ctx.clearRect(0,0,can.width,can.height);if(c.roi.length<1)return;ctx.strokeStyle='#f4c84c';ctx.lineWidth=3;ctx.beginPath();c.roi.forEach((p,i)=>{let x=p[0]*can.width,y=p[1]*can.height;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});if(c.roi.length>2)ctx.closePath();ctx.stroke();c.roi.forEach(p=>{ctx.fillStyle='#f4c84c';ctx.fillRect(p[0]*can.width-4,p[1]*can.height-4,8,8)})}function render(){grid.innerHTML='';config.cameras.forEach(c=>{let d=document.createElement('section');d.className='card';d.innerHTML=`<h2 class='cam-title'>${c.id.toUpperCase()}　${c.name}</h2><div class='fields'><label>攝影機名稱<input data-k='name' value='${c.name}'></label><label class='enabled'><input type='checkbox' data-k='enabled' ${c.enabled?'checked':''}>啟用此攝影機</label><label class='wide'>RTSP URL<input data-k='rtsp_url' value='${c.rtsp_url}' placeholder='rtsp://帳號:密碼@IP:554/path'></label><label>逗留警報秒數<input data-k='dwell_seconds' type='number' min='1' value='${c.dwell_seconds}'></label></div><div class='preview'><img src='/stream/${c.id}'><canvas></canvas></div><div class='roi-line'><span>ROI 點位：${c.roi.length}（至少 3 點）</span><button class='clear'>清除 ROI</button></div>`;grid.append(d);let count=d.querySelector('.roi-line span');for(const inp of d.querySelectorAll('input'))inp.oninput=()=>{c[inp.dataset.k]=inp.type==='checkbox'?inp.checked:inp.value;d.querySelector('.cam-title').textContent=`${c.id.toUpperCase()}　${c.name}`};let can=d.querySelector('canvas'),ctx=can.getContext('2d');can.width=640;can.height=360;draw(c,ctx,can);can.onclick=e=>{let r=can.getBoundingClientRect();c.roi.push([(e.clientX-r.left)/r.width,(e.clientY-r.top)/r.height]);count.textContent=`ROI 點位：${c.roi.length}（至少 3 點）`;draw(c,ctx,can)};d.querySelector('.clear').onclick=()=>{c.roi=[];count.textContent='ROI 點位：0（至少 3 點）';draw(c,ctx,can)}})}async function load(){config=await (await fetch('/api/config')).json();render()}document.querySelector('#save').onclick=async()=>{let button=document.querySelector('#save'),message=document.querySelector('#message');button.disabled=true;button.textContent='儲存中…';try{let response=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});if(!response.ok)throw Error();message.textContent='✓ 設定已儲存並套用';}catch{message.textContent='✕ 儲存失敗，請重試';}finally{button.disabled=false;button.textContent='儲存並套用設定'}};load();</script></body></html>"""


HOME = HOME.replace(
    "</style></head>",
    "</style><style>.dwell{position:absolute;bottom:28px;left:10px;color:#ffe17a;text-shadow:0 1px 4px #000;font-size:10px;font-weight:bold}</style></head>",
    1,
).replace(
    "<span class='camname'>${id}</span><span class='live off'>",
    "<span class='camname'>${id}</span><span class='dwell'>ROI 內無人</span><span class='live off'>",
    1,
).replace(
    "function beep(){if(!audioContext)return;for(let i=0;i<3;i++){const o=audioContext.createOscillator(),g=audioContext.createGain(),t=audioContext.currentTime+i*.22;o.frequency.value=880;g.gain.setValueAtTime(.14,t);g.gain.exponentialRampToValueAtTime(.001,t+.18);o.connect(g).connect(audioContext.destination);o.start(t);o.stop(t+.18)}}",
    "function beep(){if(!audioContext)return;const o=audioContext.createOscillator(),g=audioContext.createGain(),t=audioContext.currentTime;o.type='sawtooth';g.gain.setValueAtTime(.001,t);g.gain.linearRampToValueAtTime(.16,t+.04);for(let i=0;i<5;i++){let a=t+.04+i*.55;o.frequency.setValueAtTime(620,a);o.frequency.linearRampToValueAtTime(1180,a+.26);o.frequency.linearRampToValueAtTime(620,a+.52)}g.gain.exponentialRampToValueAtTime(.001,t+2.8);o.connect(g).connect(audioContext.destination);o.start(t);o.stop(t+2.82)}",
    1,
).replace(
    "const live=e.querySelector('.live');live.textContent=",
    "const dwell=e.querySelector('.dwell'),active=c.tracks.filter(x=>!x.alerted);dwell.textContent=active.length?`ROI 內停留 ${Math.max(...active.map(x=>x.seconds)).toFixed(1)} 秒`:'ROI 內無人';const live=e.querySelector('.live');live.textContent=",
    1,
)
SETTINGS = SETTINGS.replace("ctx.lineWidth=3", "ctx.lineWidth=5")
HOME = HOME.replace(
    "</script></body>",
    """const soundButton=document.querySelector('#sound');function enableWarningSound(){if(!audioContext)audioContext=new AudioContext();audioContext.resume().then(()=>{soundButton.classList.add('on');soundButton.innerHTML='<span class=\"dot\">●</span> 告警聲已啟用';}).catch(()=>{});}enableWarningSound();soundButton.addEventListener('click',enableWarningSound);document.addEventListener('pointerdown',enableWarningSound,{once:true});</script></body>""",
    1,
)
HOME = HOME.replace(
    "</body>",
    """<style>#alarm-dialog{display:none;position:fixed;inset:0;z-index:100;background:#000a;align-items:center;justify-content:center}.alarm-box{width:330px;border:2px solid #ef5350;background:#250d0d;box-shadow:0 0 28px #e53935aa;padding:24px;text-align:center;color:#fff}.alarm-box h2{margin:0 0 10px;color:#ff7770;font-size:18px}.alarm-box p{color:#ffd0ce;line-height:1.6}.alarm-box button{border:1px solid #ff8984;background:#bd2925;color:#fff;padding:11px 20px;font:inherit;font-weight:bold;cursor:pointer}.alarm-box button:hover{background:#dc3b36}#alarm-dialog.show{display:flex}</style><div id='alarm-dialog' role='alertdialog' aria-modal='true' aria-labelledby='alarm-title'><section class='alarm-box'><h2 id='alarm-title'>⚠ 人員逗留警報</h2><p>防空警報將持續 30 秒。<br>確認後可停止本次警報聲。</p><button id='stop-alarm'>確認關閉本次警報</button></section></div><script>let alarmOscillator=null;function stopCurrentAlarm(){if(alarmOscillator){alarmOscillator.stop();alarmOscillator=null;}document.querySelector('#alarm-dialog').classList.remove('show');}beep=function(){if(!audioContext)return;stopCurrentAlarm();const o=audioContext.createOscillator(),g=audioContext.createGain(),t=audioContext.currentTime;o.type='sawtooth';g.gain.setValueAtTime(.18,t);for(let i=0;i<55;i++){const a=t+i*.55;o.frequency.setValueAtTime(620,a);o.frequency.linearRampToValueAtTime(1180,a+.26);o.frequency.linearRampToValueAtTime(620,a+.52);}o.connect(g).connect(audioContext.destination);alarmOscillator=o;o.onended=()=>{if(alarmOscillator===o){alarmOscillator=null;document.querySelector('#alarm-dialog').classList.remove('show');}};document.querySelector('#alarm-dialog').classList.add('show');o.start(t);o.stop(t+30);};document.querySelector('#stop-alarm').addEventListener('click',stopCurrentAlarm);</script></body>""",
    1,
)
HOME = HOME.replace("active=c.tracks.filter(x=>!x.alerted)", "active=c.tracks", 1)
HOME = HOME.replace(
    "<button id='stop-alarm'>確認關閉本次警報</button>",
    "<div class='alarm-actions'><button id='stop-alarm'>確認關閉本次警報</button><button id='ignore-alarm'>忽略警報 10分鐘</button></div>",
    1,
).replace(
    "let alarmOscillator=null;function stopCurrentAlarm()",
    "let alarmOscillator=null,ignoreAlertUntil=Number(localStorage.getItem('yolo-cam-alert-muted-until')||0);function stopCurrentAlarm()",
    1,
).replace(
    "events=events.slice(0,8);beep();drawEvents()",
    "events=events.slice(0,8);if(Date.now()>=ignoreAlertUntil)beep();drawEvents()",
    1,
).replace(
    "document.querySelector('#stop-alarm').addEventListener('click',stopCurrentAlarm);",
    "document.querySelector('#stop-alarm').addEventListener('click',stopCurrentAlarm);document.querySelector('#ignore-alarm').addEventListener('click',()=>{ignoreAlertUntil=Date.now()+600000;localStorage.setItem('yolo-cam-alert-muted-until',String(ignoreAlertUntil));stopCurrentAlarm();});",
    1,
).replace(
    "</head>",
    "<style>.alarm-actions{display:flex;gap:8px;justify-content:center}.alarm-actions button{margin:0}.alarm-actions #ignore-alarm{background:#47545a;border-color:#84949a}.alarm-actions #ignore-alarm:hover{background:#5b6a70}</style></head>",
    1,
)

HOME = HOME.replace(
    "</head>",
    "<style>.sound-toggle{margin-left:13px;display:flex;align-items:center;gap:7px;color:#9bb0b4;font-size:10px;cursor:pointer}.sound-toggle input{position:absolute;opacity:0;pointer-events:none}.switch{width:31px;height:17px;border-radius:10px;background:#45545a;position:relative;transition:.15s}.switch:after{content:'';position:absolute;width:13px;height:13px;top:2px;left:2px;border-radius:50%;background:#b9c6c8;transition:.15s}.sound-toggle input:checked+.switch{background:#087f76}.sound-toggle input:checked+.switch:after{left:16px;background:#4ef0dc}.sound-toggle input:focus-visible+.switch{outline:2px solid #d7fffb;outline-offset:2px}</style></head>",
    1,
).replace(
    "</body>",
    """<script>const oldSoundControl=document.querySelector('#sound');oldSoundControl.outerHTML='<label class="sound-toggle" aria-label="告警聲開關"><input id="sound-toggle" type="checkbox"><span class="switch"></span><span class="sound-label"></span></label>';const soundToggle=document.querySelector('#sound-toggle'),soundLabel=document.querySelector('.sound-label'),savedSoundState=localStorage.getItem('yolo-cam-sound-enabled');soundToggle.checked=savedSoundState!=='false';const alarmBeep=beep;beep=()=>{if(soundToggle.checked)alarmBeep();};function updateSoundToggle(){localStorage.setItem('yolo-cam-sound-enabled',String(soundToggle.checked));soundLabel.textContent=soundToggle.checked?'告警聲已啟用':'告警聲已關閉';if(soundToggle.checked){if(!audioContext)audioContext=new AudioContext();audioContext.resume();}}soundToggle.addEventListener('change',updateSoundToggle);updateSoundToggle();</script></body>""",
    1,
)
HOME = HOME.replace(
    "</head>",
    "<style>#alarm-dialog .alarm-box{width:min(560px,calc(100vw - 32px));padding:24px 26px}.alarm-actions{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;width:100%;margin-top:18px}.alarm-actions button{min-width:0;min-height:64px;padding:12px 10px;font-size:14px;line-height:1.35;white-space:nowrap}.alarm-actions #ignore-alarm{background:#47545a;border-color:#84949a}@media (max-width:500px){.alarm-actions{grid-template-columns:1fr}.alarm-actions button{white-space:normal}}</style></head>",
    1,
)
HOME = HOME.replace("<span class='camtime'></span>", "", 1).replace(
    "e.querySelector('.camtime').textContent=new Date().toLocaleString('sv-SE');",
    "",
    1,
)
HOME = HOME.replace(
    "</body>",
    """<script>const temporarySoundToggle=document.querySelector('#sound-toggle'),temporarySoundLabel=document.querySelector('.sound-label');let soundMuteUntil=Number(localStorage.getItem('yolo-cam-sound-muted-until')||0);if(!soundMuteUntil&&localStorage.getItem('yolo-cam-sound-enabled')==='false'){soundMuteUntil=Date.now()+600000;localStorage.setItem('yolo-cam-sound-muted-until',String(soundMuteUntil));}let soundMuteTimer;function renderSoundMute(){clearTimeout(soundMuteTimer);if(soundMuteUntil>Date.now()){temporarySoundToggle.checked=false;const when=new Date(soundMuteUntil);temporarySoundLabel.textContent=`告警聲暫停，${when.toLocaleTimeString('zh-TW',{hour:'2-digit',minute:'2-digit'})} 自動啟用`;soundMuteTimer=setTimeout(renderSoundMute,1000);}else{soundMuteUntil=0;localStorage.removeItem('yolo-cam-sound-muted-until');localStorage.setItem('yolo-cam-sound-enabled','true');temporarySoundToggle.checked=true;temporarySoundLabel.textContent='告警聲已啟用';}}temporarySoundToggle.addEventListener('change',()=>{if(temporarySoundToggle.checked){soundMuteUntil=0;localStorage.removeItem('yolo-cam-sound-muted-until');localStorage.setItem('yolo-cam-sound-enabled','true');if(!audioContext)audioContext=new AudioContext();audioContext.resume();}else{soundMuteUntil=Date.now()+600000;localStorage.setItem('yolo-cam-sound-muted-until',String(soundMuteUntil));localStorage.setItem('yolo-cam-sound-enabled','false');}renderSoundMute();});renderSoundMute();</script></body>""",
    1,
)


def create_app(settings: Settings) -> Flask:
    store = CameraConfigStore(settings.camera_config_path, settings)
    service = MonitorService(settings, store)
    app = Flask(__name__)

    @app.get("/")
    def home():
        return render_template_string(HOME)

    @app.get("/settings")
    def settings_page():
        return render_template_string(SETTINGS)

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.route("/api/config", methods=["GET", "POST"])
    def config():
        if request.method == "GET":
            return jsonify(store.get())
        try:
            saved = store.save(request.get_json(force=True))
            service.apply(saved)
            return jsonify(saved)
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/stream/<camera_id>")
    def stream(camera_id: str):
        return Response(service.stream(camera_id), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def run_dashboard() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    settings = Settings.from_env()
    _configure_logging(settings.log_level)
    LOGGER.info("Starting web control room at http://%s:%d", settings.web_host, settings.web_port)
    create_app(settings).run(host=settings.web_host, port=settings.web_port, threaded=True)

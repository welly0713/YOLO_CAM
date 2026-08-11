# YOLO_CAM

## 先用電腦鏡頭測試

目前的 `.env` 已經預設使用本機鏡頭 `index=0`，並開啟即時預覽。直接安裝套件後執行：

```powershell
python -m pip install -r requirements.txt
python -m app.main
```

預覽視窗會顯示 YOLO 偵測結果與目前 15 秒視窗統計。按 `q` 或 `Esc` 可以停止；也可以按 `Ctrl+C`。如果電腦有多個鏡頭，將 `.env` 的 `WEBCAM_INDEX` 改成 `1`、`2` 等索引。

要切回 RTSP，將 `.env` 改為：

```dotenv
CAMERA_SOURCE=rtsp
SHOW_PREVIEW=false
```

本機鏡頭測試建議直接在 Windows 主機執行；Docker 主要保留給 RTSP，因為 Docker Desktop 不會自動把 Windows 本機鏡頭傳入容器。

用 Python + OpenCV + Ultralytics YOLO 讀取 RTSP 監視器，偵測畫面中是否有人。

目前版本的告警會輸出到 CMD，不會寄信；SMTP 設定已經放進 `.env`，之後只要打開開關即可使用。

## 偵測規則

程式預設每 1 秒從 RTSP 串流取最新畫面並執行一次 YOLO。每次只計算 COCO 的 `person` 類別：

```text
最近 15 次取樣中，至少 10 次偵測到人（10 / 15 = 2 / 3）=> 輸出一則告警
```

視窗是滑動視窗。為避免同一個人停留時每秒重複告警，事件仍持續時只告警一次；當視窗不再符合條件後，下一個新事件才會再次告警。`ALERT_COOLDOWN_SECONDS` 也會限制兩個事件之間的最短告警間隔。

## 本機執行

1. 複製設定檔並填入監視器資訊：

   ```powershell
   Copy-Item .env.example .env
   ```

   編輯 `.env` 的 `RTSP_HOST`、`RTSP_PORT`、`RTSP_USERNAME`、`RTSP_PASSWORD`、`RTSP_PATH`。不同品牌的 RTSP path 不同，常見格式例如 `/Streaming/Channels/101`、`/h264Preview_01_main`；請以監視器手冊為準。

2. 建立虛擬環境並安裝套件：

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

3. 啟動：

   ```powershell
   python -m app.main
   ```

第一次啟動如果本機沒有 `yolo26n.pt`，Ultralytics 會下載模型；需要能連網一次。可用 `Ctrl+C` 停止。

## Docker 執行

先確認 `.env` 已建立且設定完成，再執行：

```powershell
docker compose up --build
```

背景執行：

```powershell
docker compose up --build -d
docker compose logs -f yolo-cam
```

## Web 中控台

Docker 啟動後，在中控室電腦瀏覽器開啟：

```text
http://Docker主機IP:8090
```

首頁同時顯示三台即時監視器畫面與頂端連線狀態。YOLO 對每台已啟用攝影機只會每秒推論一次，畫面串流不受此限制。右上角按「啟用告警聲」後，符合 ROI 逗留條件的事件會框紅畫面並播放聲音。

按「設定攝影機與 ROI」可設定四台 IP Camera 的 RTSP URL、啟用狀態、逗留秒數，以及用滑鼠在預覽畫面圈選 ROI。ROI 至少需要三個點；同一人持續在該區域達設定秒數才會警報。設定檔會保存於 Docker volume，不會因容器重建而遺失。

Docker 會將 Ultralytics 模型快取保存於 named volume，因此重建容器不必重複下載。若攝影機在區域網路中，Docker Desktop 通常可以直接連到區域網路 IP；若仍連不上，先在主機測試 RTSP URL，再檢查防火牆與攝影機是否允許該主機連線。

## `.env` 重要設定

| 設定 | 預設值 | 說明 |
|---|---:|---|
| `RTSP_URL` | 空白 | 可直接指定完整 RTSP URL；有值時優先於分開的 RTSP 欄位 |
| `CAMERA_SOURCE` | `rtsp` | `rtsp` 使用監視器；`webcam` 使用本機鏡頭 |
| `WEBCAM_INDEX` | `0` | 本機鏡頭索引 |
| `WEBCAM_WIDTH` / `WEBCAM_HEIGHT` | `1280` / `720` | 本機鏡頭要求的解析度 |
| `SHOW_PREVIEW` | `false` | 是否顯示 OpenCV 即時預覽視窗 |
| `RTSP_HOST` / `RTSP_PORT` | — / `554` | 監視器 IP 與 RTSP port |
| `RTSP_USERNAME` / `RTSP_PASSWORD` | — | RTSP 帳號與密碼 |
| `RTSP_PATH` | `/Streaming/Channels/101` | 監視器的串流路徑 |
| `MODEL_PATH` | `yolo26n.pt` | YOLO 模型名稱或容器內的模型檔案路徑 |
| `PERSON_CONFIDENCE` | `0.5` | 人員偵測信心度門檻，0 到 1 |
| `SAMPLE_INTERVAL_SECONDS` | `1` | 取樣間隔 |
| `DETECTION_WINDOW_SECONDS` | `15` | 滑動判斷視窗 |
| `DETECTION_MIN_RATIO` | `0.6666667` | 視窗內至少多少比例的取樣有人 |
| `ALERT_COOLDOWN_SECONDS` | `300` | 事件間隔的最短告警時間 |
| `EMAIL_ENABLED` | `false` | 設成 `true` 後，告警會在 CMD 輸出以外透過 SMTP 寄出 |
| `SMTP_*` / `EMAIL_*` | 範例值 | 未來寄信所需的 SMTP 與收件人資料 |

你提到的 email「sftp」應該是 SMTP（寄信協定）；本專案先依 SMTP 預留欄位。啟用前請把 `SMTP_HOST`、`SMTP_PORT`、帳號、密碼、`EMAIL_FROM`、`EMAIL_TO` 換成實際資料。

## 測試

不需要攝影機或 YOLO 模型即可測試 15 秒判斷邏輯：

```powershell
pytest -q
```

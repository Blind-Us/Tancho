# 任務：Tancho 實機（Raspberry Pi 4）— 校正嘗試 + 神經網路推論節點（不驅動馬達）

你在 Tancho 輪腿機器人的 Raspberry Pi 4（8 GB，ARM64）上工作，擁有完整權限。這台 Pi 直接連著真實的馬達和 IMU，**操作錯誤會讓機器人突然動作、損壞硬體或傷到人**，所以下面的「邊界」優先於任何其他考量。遇到邊界外的事情，就停下來寫進報告，不要自己想辦法繞過去。

回報一律用繁體中文。

---

## 0. 背景

- **機器人**：兩條腿，每腿 DM4310 大腿、DM4310 小腿、DMH3510 輪子；IMU 是達妙 DM-IMU-L1（USB）。
- **現有的只讀節點**（你之前寫的）：`/home/tancho/Documents/Codex/2026-10-04/ehl/outputs/tancho_dds/`
  - 啟動指令 `tancho-state`；DDS domain 47、topic `tancho/hardware_state`、100 Hz。
  - 它不會 enable 馬達。上次驗證的結果：
    - `ch0_id2`、`ch0_id3` 沒有回應；
    - `ch1_id2` 狀態碼 3；
    - `mapping_confirmed`、`imu_axes_confirmed` 都是 false，IMU 單位還沒確認。
- **policy（walk v1）**：https://github.com/Blind-Us/Tancho/releases/tag/walk-v1
  - `policy.onnx`：輸入 `obs` float32 [1,25]，輸出 `actions` float32 [1,6]。
  - `DEPLOY.md`：**觀測和動作的唯一規格來源**，動手前先完整讀過。
  - `reference_io.json`：10 組「觀測 → 動作」的參考答案，容許誤差 1e-4。
  - `SHA256SUMS`：下載後先 `sha256sum -c SHA256SUMS`，不通過就停。

---

## 1. 邊界（絕對禁止，沒有例外）

### 馬達
- **不准呼叫**以下任何會讓馬達動作或改變馬達狀態的函式：
  - `enable`、`enable_all`、`Hardware.enable`；
  - `control_mit`、`control_vel`、`control_pos_vel`；
  - `Hardware.apply`。
- **不准送出**設零點（`FE`）、清錯誤、改 CAN ID / Master ID、改波特率、改控制模式，或寫入任何馬達參數暫存器的命令；**不准刷韌體**。
- 不准修改 `hardware.py`、`damiao.py` 裡和送命令有關的程式，也不准為了測試而寫一個會送命令的版本。
- 這一輪的推論節點**只產生數字、不送命令**。程式裡不得 import 或持有任何能送 CAN 命令的物件。

### 裝置和系統
- 推論節點只透過 DDS 訂閱 `tancho/hardware_state`，**不直接開 CAN 或 IMU 裝置**，避免和 `tancho-state` 搶裝置。
- 不准改 udev 規則、systemd、開機自動啟動、網路、防火牆、SSH 設定；不准用 `sudo` 做系統層級的變更。
- 套件只能裝進既有的 `tancho-pi` conda 環境，裝了什麼要列在報告裡。
- 不准刪除或覆寫既有的 gateway、controller、state node 檔案。新程式一律放在新目錄 `.../outputs/tancho_policy/`。
- 不准修改 `policy.onnx`。

### 校正
- 只能在**馬達未使能**的狀態下做，全部是讀取：
  - 用手轉動關節、傾斜機身，看讀值怎麼變。
  - 需要有人動手的步驟，寫清楚請使用者做什麼，等使用者回覆後再繼續。
- `mapping_confirmed`、`imu_axes_confirmed` 只有在**每一項都有實測紀錄**時才能改成 true。缺任何一項就維持 false。
- 不要猜。「依照排列順序推測」不算確認。

### 什麼時候停下來
遇到以下任何一種情況，停下來寫報告，不要自行處理：
- 需要讓馬達通電才能繼續；
- 需要 sudo 或系統變更；
- 規格（`DEPLOY.md`）和實機對不上，又無法確定該怎麼對應；
- 任何你不確定會不會讓機器人動的操作。

---

## 2. 任務 A：嘗試校正（做不到就跳過，不要卡住）

目標是填好 `hardware-state.yaml` 的對應、方向和零點，以及 IMU 的單位和軸向。**沒回應的馬達（`ch0_id2`、`ch0_id3`）和狀態碼 3（`ch1_id2`）不要嘗試修**，記在報告裡就好。

1. **馬達對應**：請使用者一次只轉一個關節，記下哪個 `chX_idY` 的位置在變，填入 `role`：
   `thigh_L`、`calf_L`、`wheel_L`、`thigh_R`、`calf_R`、`wheel_R`。
   左右的定義：機器人面向前方時，它自己的左和右。
2. **方向**：比對 `DEPLOY.md` 和模擬的正方向：
   - 把腿往前抬時，大腿讀值的變化方向；
   - 輪子往前滾時，輪速的正負號。

   以模擬為準決定 `direction`。如果無法確定模擬的正方向，就把實測現象寫下來，留給使用者判斷。
3. **零點**：
   - 模型的標稱腿角是 thigh −0.50、calf +0.87 rad。
   - 把腿擺到可以重現的已知姿勢（例如機械限位），記錄讀值。
   - 偏移量寫在軟體設定裡，格式是 `q_model = direction × q_motor + offset`。
   - **不准用 `FE` 命令把零點寫進馬達**。
   - 如果沒有可靠的已知姿勢，就跳過這一步並在報告裡說明。
4. **IMU 單位**：
   - 直立靜止時，加速度的大小接近 9.81 就是 m/s²，接近 1.0 就是 g。
   - 陀螺儀：把機身繞一個軸轉一個已知角度，將角速度積分，和 RPY 的變化比對，算出 rad/s 的換算比例。
5. **IMU 軸向**：模型要的座標是 **x 朝機身前方、y 朝上、z 朝機身右方**。
   - 直立靜止時，重力讀值應該是 (0, −1, 0)；機身往前傾（低頭）時，x 分量會變正。
   - pitch 角速度在 z 軸。
   - 找出把 IMU 本體座標轉到模型座標的 3×3 旋轉矩陣，並用「直立、前傾、右傾」三個姿勢驗證。

---

## 3. 任務 B：神經網路推論節點（dry-run，不送命令）

新目錄：`/home/tancho/Documents/Codex/2026-10-04/ehl/outputs/tancho_policy/`

### 3.1 推論
- 優先用 `onnxruntime`（ARM64、Python 3.13）。裝不起來的話，用 numpy 從 ONNX 檔讀出權重自己算。這個網路是小型 MLP（ELU 激活），觀測正規化已經包在模型裡。
- **先通過 `reference_io.json` 的 10 組測試**，每個輸出的絕對誤差都要 < 1e-4，才能往下做。

### 3.2 組觀測（50 Hz），規格以 `DEPLOY.md` 為準

| index | 來源 | 處理 |
|---|---|---|
| 0–2 | IMU 四元數 | 算出 IMU 本體座標的重力單位向量，再乘上任務 A 的旋轉矩陣，轉到模型座標 |
| 3–5 | IMU 角速度 | 換成 rad/s → 轉到模型座標 → × 0.25 |
| 6–8 | 命令 | 這一輪固定 (0, 0, 0)；另外留一個 DDS topic 或參數介面，供之後接搖桿 |
| 9–10 | 輪速 L、R | rad/s，乘上 `direction`，× 0.05 |
| 11–16 | 上一步 clip 後的 action | 開機時為 0 |
| 17–20 | 腿角 thigh_L, calf_L, thigh_R, calf_R | `q_model − 標稱`（−0.50 / +0.87） |
| 21–24 | 腿角速度，順序同上 | × 0.05 |

- **注意關節順序**：thigh_L, calf_L, thigh_R, calf_R。這和馬達通道、ID 的排列順序不一樣，一定要用任務 A 的 `role` 對應，不能照排列順序填。

### 3.3 輸出（只發布、不送）
- `actions` 先 clip 到 [−1, 1]，再換算：
  - 腿：目標角 = 標稱 + 0.25·a；
  - 輪：目標輪速 = 52.36·a rad/s。
- 發布到新的 DDS topic `tancho/policy_preview`（domain 47），內容包括：
  - 時間戳、25 維觀測、6 維原始和 clip 後的 action；
  - 腿目標角、輪目標速度；
  - 一個 `would_command: false` 欄位；
  - 診斷訊息。
- **安全旗標**：以下任一情況成立時，`safe_to_command = false`，並寫明原因：
  - 有馬達資料缺失或過期（> 150 ms）；
  - 有馬達狀態碼不是 0 或 1；
  - IMU 過期；
  - `mapping_confirmed` 或 `imu_axes_confirmed` 為 false；
  - IMU 單位未確認；
  - 傾角 > 15°；
  - 觀測裡有 NaN。

  這一輪無論如何都不會送命令，這個旗標是留給下一輪用的。
- 用 `--log` 參數記錄 JSON lines，供之後和模擬對照。

### 3.4 驗證
1. 離線：`reference_io.json` 的 10 組全部通過。
2. 實機只讀：`tancho-state` 和推論節點同時跑 30 秒，機器人靜止直立（馬達未使能）。記錄：
   - 頻率是否穩定在 50 Hz；
   - 推論延遲（平均和最大）；
   - 觀測 0–2 是否接近 (0, −1, 0)；
   - `safe_to_command` 的值和原因。
3. 用手前傾、後傾機身，以及轉動腿，確認觀測各欄位的正負號和 `DEPLOY.md` 一致。

---

## 4. 交付報告（繁體中文）

1. **校正表**：每顆馬達的 `chX_idY` → `role`、`direction`、`offset`，並標明每一項是「實測確認」還是「未確認」。
2. **IMU**：單位、旋轉矩陣，以及三個姿勢的驗證讀值。
3. **推論節點**：
   - 檔案清單、啟動指令、安裝的套件；
   - `reference_io.json` 的測試結果；
   - 實機 30 秒的頻率和延遲。
4. **安全旗標**：目前 `safe_to_command` 的值和所有原因。
5. **未解決的問題**：沒回應的兩顆馬達、狀態碼 3，以及任何規格對不上的地方。
6. **聲明**：確認整個過程中沒有呼叫任何 enable 或 control 函式，沒有送出任何馬達命令。
